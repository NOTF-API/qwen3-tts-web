"""One model owner, bounded admission and memory-aware native batch inference.

Do not run concurrent generate() calls on a shared Transformers model. Pending
requests are combined into one native batch; prompt creation and model switching
share the same lock. No torch import is needed for queue/policy unit tests.
"""

import gc
import logging
import math
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field

GIB = 1024**3
log = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    def __init__(self, detail, status_code=503):
        super().__init__(detail)
        self.status_code = status_code


@dataclass
class Job:
    key: str
    text: str
    language: str
    prompt: object = None
    instruct: str = ""
    allow_download: bool = False
    clip_id: str | None = None
    future: Future = field(default_factory=Future)
    generation_stats: dict = field(default_factory=dict)

    @property
    def weight(self):
        # A conservative length proxy, including reference audio/text. Native
        # batches pad to their longest member, so admission uses max, not sum.
        reference = getattr(self.prompt, "ref_code", None)
        frames = int(reference.shape[0]) if reference is not None else 0
        length = len(self.text) + len(self.instruct) + len(getattr(self.prompt, "ref_text", "") or "") + frames * 4
        return 2 ** max(0, math.ceil(math.log2(max(1, length / 128))))

    @property
    def group(self):
        return self.key, self.allow_download


@dataclass(frozen=True)
class MemorySnapshot:
    total: int
    free: int
    allocated: int
    reserved: int

    def budget(self, settings):
        # Cached blocks are reusable by this process and are not driver-free.
        reusable = max(0, self.reserved - self.allocated)
        reserve = max(settings.gpu_memory_reserve_gb * GIB, self.total * (1 - settings.gpu_memory_fraction))
        return max(0, min(
            self.free + reusable - reserve,
            self.total * settings.gpu_memory_fraction - self.allocated,
        ))


class DeviceMemory:
    def __init__(self, device, settings):
        import torch

        self.torch = torch
        self.device = str(device)
        self.cuda = self.device.startswith("cuda")
        self.settings = settings

    def snapshot(self):
        if not self.cuda:
            return None
        try:
            free, total = self.torch.cuda.mem_get_info(self.device)
            return MemorySnapshot(total, free,
                self.torch.cuda.memory_allocated(self.device),
                self.torch.cuda.memory_reserved(self.device))
        except (RuntimeError, ValueError):
            log.warning("无法测量 CUDA 显存，使用单条推理", exc_info=True)
            return None

    def begin(self):
        if not self.cuda:
            return 0
        self.torch.cuda.synchronize(self.device)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        return self.torch.cuda.memory_allocated(self.device)

    def peak(self):
        return self.torch.cuda.max_memory_allocated(self.device) if self.cuda else 0

    def is_oom(self, exc):
        return isinstance(exc, self.torch.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )

    def recover(self):
        # Called outside the exception handler: its traceback must no longer
        # retain generation tensors before garbage collection / empty_cache.
        gc.collect()
        if self.cuda:
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()
        elif self.device.startswith("mps"):
            self.torch.mps.empty_cache()

    def release_idle_cache(self):
        # Keep model tensors resident; only return unused CUDA allocator blocks.
        # MPS retains its existing memory-management behavior.
        if self.cuda:
            gc.collect()
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()


@dataclass
class Profile:
    bytes_per_weight: float = 1.5 * GIB
    proven_batch: int = 0
    oom_limit: int = 64
    successes_after_oom: int = 0


class BatchPolicy:
    def __init__(self, settings):
        self.settings = settings
        self.profiles = {}

    def profile(self, job):
        # Keep long references / long instructions from borrowing a short-text
        # calibration, and keep Base and VoiceDesign measurements independent.
        return self.profiles.setdefault((job.key, job.weight), Profile())

    def capacity(self, jobs, snapshot):
        if snapshot is None:
            return 1
        profile = self.profile(max(jobs, key=lambda j: j.weight))
        if not profile.proven_batch:
            return 1  # first request for this length class calibrates alone
        cost = profile.bytes_per_weight * max(j.weight for j in jobs)
        by_memory = int(snapshot.budget(self.settings) / max(1, cost))
        return max(1, min(self.settings.max_batch_size, by_memory,
                          profile.proven_batch * 2, profile.oom_limit))

    def succeeded(self, jobs, incremental_peak):
        weight = max(j.weight for j in jobs)
        measured = max(0.5 * GIB, incremental_peak / len(jobs) / weight * 1.5)
        for job in jobs:
            profile = self.profile(job)
            # Never erase a higher observation with an unusually short output.
            profile.bytes_per_weight = max(measured, profile.bytes_per_weight if profile.proven_batch else 0)
            profile.proven_batch = max(profile.proven_batch, len(jobs))
        for key in {(j.key, j.weight) for j in jobs}:
            profile = self.profiles[key]
            profile.successes_after_oom += 1
            if profile.successes_after_oom >= 4:
                profile.oom_limit = min(64, profile.oom_limit * 2)
                profile.successes_after_oom = 0

    def failed(self, jobs):
        for key in {(j.key, j.weight) for j in jobs}:
            profile = self.profiles.setdefault(key, Profile())
            profile.bytes_per_weight *= 1.5
            profile.oom_limit = max(1, len(jobs) // 2)
            profile.successes_after_oom = 0


class InferenceScheduler:
    def __init__(self, settings, model_loader, model_lock, memory_factory=DeviceMemory, idle_cache_seconds=10):
        self.settings = settings
        self.model_loader = model_loader
        self.model_lock = model_lock
        self.memory_factory = memory_factory
        self.policy = BatchPolicy(settings)
        self.condition = threading.Condition()
        self.pending = deque()
        self.active = []
        self.owned = []
        self.closed = False
        self.completed = 0
        self.oom_retries = 0
        self.last_batch_size = 0
        self.last_peak_bytes = 0
        self.last_seconds = 0.0
        self.last_capacity = 1
        self.last_memory = None
        self.active_phase = "idle"
        self.active_started = None
        self.batch_sequence = 0
        self.idle_memory = None
        self.idle_cache_seconds = idle_cache_seconds
        self.worker = threading.Thread(target=self._work, name="tts-inference", daemon=True)

    def start(self):
        self.worker.start()

    def submit(self, job):
        with self.condition:
            if self.closed:
                raise InferenceError("服务正在关闭，请稍后重试")
            # Reap cancelled callers before applying the admission limit.
            self.pending = deque(j for j in self.pending if not j.future.cancelled())
            if len(self.pending) + sum(not j.future.done() for j in self.owned) >= self.settings.max_pending_jobs:
                raise InferenceError("生成队列已满，请稍后重试", 429)
            self.pending.append(job)
            self.condition.notify()
        return job.future

    def status(self):
        with self.condition:
            active_ids = {id(j) for j in self.active}
            queued = [j for j in self.pending if not j.future.cancelled()]
            queued += [j for j in self.owned if id(j) not in active_ids and not j.future.done()]
            return {
                "mode": "adaptive_batch", "max_batch_size": self.settings.max_batch_size,
                "batch_capacity": self.last_capacity, "queued": len(queued),
                "active": len(self.active),
                "active_phase": self.active_phase,
                "active_batch_id": self.batch_sequence if self.active_started is not None else None,
                "active_elapsed_seconds": round(time.monotonic() - self.active_started, 1) if self.active_started is not None else 0,
                "active_clip_ids": [j.clip_id for j in self.active if j.clip_id],
                "queued_clip_ids": [j.clip_id for j in queued if j.clip_id],
                "completed": self.completed, "oom_retries": self.oom_retries,
                "last_batch_size": self.last_batch_size,
                "last_peak_gb": round(self.last_peak_bytes / GIB, 3),
                "last_batch_seconds": round(self.last_seconds, 3),
                "memory": self.last_memory,
            }

    def close(self):
        with self.condition:
            self.closed = True
            while self.pending:
                job = self.pending.popleft()
                if job.future.set_running_or_notify_cancel():
                    job.future.set_exception(InferenceError("服务正在关闭"))
            self.condition.notify_all()
        if self.worker.ident is not None:
            self.worker.join()  # never free the model while generate is running

    def _work(self):
        while True:
            with self.condition:
                ready = self.condition.wait_for(lambda: self.closed or self.pending,
                    self.idle_cache_seconds if self.idle_memory is not None else None)
                if self.closed:
                    return
            if not ready:
                self._release_idle_cache()
                continue
            with self.condition:
                # Small collection window also covers requests from other tabs.
                self.condition.wait_for(lambda: self.closed, self.settings.batch_wait_ms / 1000)
                if self.closed:
                    return
                first = self.pending.popleft()
                if not first.future.set_running_or_notify_cancel():
                    continue
                self.active = [first]
                self.owned = [first]
                self.active_phase = "preparing"
            batch = [first]
            try:
                with self.model_lock:
                    self._process(first, batch)
            except Exception as exc:
                # A model load failure must not kill the worker or leave waiters
                # unresolved. Do not retain tracebacks in request futures.
                log.exception("推理任务失败")
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(InferenceError(
                            str(getattr(exc, "detail", exc)), getattr(exc, "status_code", 500)))
            finally:
                with self.condition:
                    self.active = []
                    self.owned = []
                    self.active_phase = "idle"
                    self.active_started = None
                batch.clear()
                first = None

    def _process(self, first, batch):
        # Keep the model reference in this short-lived frame. Retaining `target`
        # in the worker loop would keep the old model alive during the next swap.
        target = self.model_loader(first.key, first.allow_download)
        memory = self.memory_factory(target.device, self.settings)
        snapshot = memory.snapshot()
        with self.condition:
            while self.pending and len(batch) < self.settings.max_batch_size:
                other = self.pending[0]
                if other.future.cancelled():
                    self.pending.popleft()
                    continue
                # FIFO across models and length classes avoids both starvation
                # and padding a short clip to the length of a huge one.
                if other.group != first.group or other.weight != first.weight:
                    break
                if len(batch) + 1 > self.policy.capacity(batch + [other], snapshot):
                    break
                self.pending.popleft()
                if other.future.set_running_or_notify_cancel():
                    batch.append(other)
            self.active = list(batch)
            self.owned = list(batch)
        try:
            self._execute(target, memory, batch)
        finally:
            self._record_memory(memory)
            with self.condition:
                self.idle_memory = memory if memory.cuda else None

    def _record_memory(self, memory):
        snapshot = memory.snapshot()
        if snapshot:
            with self.condition:
                self.last_memory = {
                    "kind": "cuda",
                    "total_gb": round(snapshot.total / GIB, 2),
                    "free_gb": round(snapshot.free / GIB, 2),
                    "allocated_gb": round(snapshot.allocated / GIB, 3),
                    "reserved_gb": round(snapshot.reserved / GIB, 3),
                    "cached_gb": round(max(0, snapshot.reserved - snapshot.allocated) / GIB, 3),
                    "budget_gb": round(snapshot.budget(self.settings) / GIB, 2),
                }

    def _release_idle_cache(self):
        # The same lock excludes prompt creation/model switching. Do not hold
        # the condition during CUDA work, so HTTP status/submission stay fast.
        with self.model_lock:
            with self.condition:
                if self.pending or self.closed:
                    return
                memory, self.idle_memory = self.idle_memory, None
            if memory is not None:
                try:
                    memory.release_idle_cache()
                    self._record_memory(memory)
                except Exception:
                    log.warning("空闲 CUDA 缓存释放失败，后续生成仍可继续", exc_info=True)

    def _attempt(self, target, memory, jobs):
        try:
            baseline = memory.begin()
            started = time.monotonic()
            with self.condition:
                self.batch_sequence += 1
                self.active_started = started
                self.active_phase = "calibrating" if memory.cuda and not self.policy.profile(jobs[0]).proven_batch else "generating"
                batch_id = self.batch_sequence
                calibrating = self.active_phase == "calibrating"
            kwargs = dict(text=[j.text for j in jobs], language=[j.language for j in jobs])
            if jobs[0].key == "voicedesign":
                wavs, rate = target.generate_voice_design(**kwargs, instruct=[j.instruct for j in jobs])
            else:
                wavs, rate = target.generate_voice_clone(**kwargs, voice_clone_prompt=[j.prompt for j in jobs])
            if len(wavs) != len(jobs):
                raise RuntimeError("模型返回的音频数量与任务数量不一致")
            peak = memory.peak()
            seconds = time.monotonic() - started
            self.policy.succeeded(jobs, max(0, peak - baseline))
            with self.condition:
                self.last_batch_size = len(jobs)
                self.last_peak_bytes = peak
                self.last_seconds = seconds
                self.completed += len(jobs)
            for job, wav in zip(jobs, wavs):
                job.generation_stats = {"batch_id": batch_id, "batch_size": len(jobs),
                    "batch_seconds": round(seconds, 3), "calibrating": calibrating}
                job.future.set_result((wav, rate))
            log.info("TTS batch=%s elapsed=%.2fs peak=%.2f GiB", len(jobs), seconds, peak / GIB)
            return None
        except Exception as exc:
            oom = memory.is_oom(exc)
            if not oom:
                log.exception("批次生成失败，将隔离失败任务")
            # Strings only: a retained traceback prevents releasing GPU tensors.
            return oom, str(exc)

    def _execute(self, target, memory, jobs):
        snapshot = memory.snapshot()
        capacity = self.policy.capacity(jobs, snapshot)
        with self.condition:
            self.last_capacity = capacity
        self._record_memory(memory)
        if len(jobs) > capacity:
            for offset in range(0, len(jobs), capacity):
                self._execute(target, memory, jobs[offset:offset + capacity])
            return
        if snapshot and snapshot.budget(self.settings) < 128 * 1024**2:
            for job in jobs:
                message = "可用显存不足，已保留安全余量。请关闭其他 GPU 程序后重试。"
                job.future.set_exception(InferenceError(message))
            return
        with self.condition:
            self.active = list(jobs)
        failure = self._attempt(target, memory, jobs)
        if failure is None:
            return
        oom, detail = failure
        if oom:
            self.policy.failed(jobs)
            with self.condition:
                self.oom_retries += int(len(jobs) > 1)
            memory.recover()
            log.warning("显存不足，batch=%s，%s", len(jobs),
                        "自动缩小重试" if len(jobs) > 1 else "返回单条任务错误，保留服务")
        if len(jobs) > 1:
            middle = len(jobs) // 2
            self._execute(target, memory, jobs[:middle])
            self._execute(target, memory, jobs[middle:])
        else:
            message = "单条任务可用内存/显存仍不足，请缩短文本/参考音频、关闭其他应用，或使用 0.6B 模型后重试。" if oom else f"生成失败: {detail}"
            jobs[0].future.set_exception(InferenceError(message, 503 if oom else 500))
