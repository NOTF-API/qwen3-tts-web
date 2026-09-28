import threading
import time
import unittest
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qwen3_tts_web.config import Settings, load_settings
from qwen3_tts_web.inference import (
    GIB, BatchPolicy, InferenceError, InferenceScheduler, Job, MemorySnapshot,
)
from qwen3_tts_web.runtime import apply_memory_limit


class FakeOOM(RuntimeError):
    pass


class FakeMemory:
    def __init__(self, device="cuda:0", settings=None):
        self.cuda = str(device).startswith("cuda")
        self.current = MemorySnapshot(24 * GIB, 18 * GIB, 4 * GIB, 4 * GIB)
        self.recovered = 0
        self.retained = None
        self.idle_released = threading.Event()

    def snapshot(self):
        return self.current if self.cuda else None

    def begin(self):
        return 4 * GIB

    def peak(self):
        return 5 * GIB

    def is_oom(self, exc):
        return isinstance(exc, FakeOOM)

    def recover(self):
        if self.retained is not None:
            assert self.retained() is None, "OOM traceback retained generation temporaries"
        self.recovered += 1

    def release_idle_cache(self):
        self.current = replace(self.current, reserved=self.current.allocated)
        self.idle_released.set()


class FakeModel:
    device = "cuda:0"

    def __init__(self):
        self.calls = []
        self.maximum = 100
        self.gate = None
        self.entered = threading.Event()
        self.memory = None

    def generate_voice_clone(self, text, language, voice_clone_prompt):
        self.calls.append((list(text), list(language), list(voice_clone_prompt)))
        self.entered.set()
        if self.gate:
            self.gate.wait(5)
        if len(text) > self.maximum or "oom" in text:
            held = SimplePayload()
            if self.memory:
                self.memory.retained = weakref.ref(held)
            raise FakeOOM("simulated CUDA out of memory")
        if "bad" in text:
            raise ValueError("bad prompt")
        return [f"{t}:{p}" for t, p in zip(text, voice_clone_prompt)], 24000

    def generate_voice_design(self, text, language, instruct):
        return self.generate_voice_clone(text, language, instruct)


class SimplePayload:
    pass


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(Path.cwd())
        self.job = Job("base", "短文本", "Chinese")

    def test_cuda_budget_accounts_for_external_use_and_cache(self):
        snapshot = MemorySnapshot(24 * GIB, 12 * GIB, 4 * GIB, 6 * GIB)
        self.assertAlmostEqual(snapshot.budget(self.settings) / GIB, 11.6)
        busy = replace(snapshot, free=1 * GIB)
        self.assertAlmostEqual(busy.budget(self.settings) / GIB, 0.6)

    def test_growth_is_calibrated_and_smaller_cards_get_smaller_batches(self):
        policy = BatchPolicy(self.settings)
        big = MemorySnapshot(24 * GIB, 18 * GIB, 4 * GIB, 4 * GIB)
        small = MemorySnapshot(8 * GIB, 2 * GIB, 4 * GIB, 4 * GIB)
        self.assertEqual(policy.capacity([self.job], big), 1)
        policy.succeeded([self.job], GIB)
        self.assertEqual(policy.capacity([self.job], big), 2)
        policy.succeeded([self.job] * 2, 2 * GIB)
        self.assertEqual(policy.capacity([self.job], big), 4)
        self.assertEqual(policy.capacity([self.job], small), 1)

    def test_long_text_reference_and_model_have_separate_calibration(self):
        policy = BatchPolicy(self.settings)
        big = MemorySnapshot(80 * GIB, 60 * GIB, 4 * GIB, 4 * GIB)
        policy.succeeded([self.job] * 8, GIB)
        self.assertEqual(policy.capacity([self.job], big), 16)
        self.assertEqual(policy.capacity([replace(self.job, text="长" * 2000)], big), 1)
        self.assertEqual(policy.capacity([replace(self.job, key="voicedesign")], big), 1)
        reference = SimpleNamespace(ref_code=SimpleNamespace(shape=(500, 16)), ref_text="参考")
        self.assertGreater(replace(self.job, prompt=reference).weight, self.job.weight)

    def test_mps_remains_serial_even_with_large_unified_memory(self):
        policy = BatchPolicy(self.settings)
        policy.succeeded([self.job] * 16, GIB)
        self.assertEqual(policy.capacity([self.job], None), 1)

    def test_mac_allocator_and_system_memory_are_untouched(self):
        # No mps / psutil API should be accessed, even on a 16 GB Mac with
        # little reported free RAM. Only CUDA receives a new allocator cap.
        apply_memory_limit("mps", self.settings, SimpleNamespace())

    def test_config_validation_and_environment_types(self):
        for values in ({"max_batch_size": 0}, {"batch_wait_ms": True},
                       {"gpu_memory_fraction": float("nan")}, {"gpu_memory_fraction": 1.0}):
            with self.assertRaises(ValueError):
                replace(self.settings, **values)
        with patch.dict("os.environ", {"QWEN3_TTS_MAX_BATCH_SIZE": "4", "QWEN3_TTS_GPU_MEMORY_FRACTION": "0.8"}):
            settings = load_settings()
            self.assertEqual(settings.max_batch_size, 4)
            self.assertEqual(settings.gpu_memory_fraction, 0.8)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(Path.cwd(), batch_wait_ms=0)
        self.memory = FakeMemory()
        self.model = FakeModel()
        self.model.memory = self.memory
        self.loaded = []
        self.scheduler = InferenceScheduler(self.settings, self.load, threading.Lock(), lambda *_: self.memory)

    def load(self, key, allow):
        self.loaded.append((key, allow))
        return self.model

    def tearDown(self):
        if self.model.gate:
            self.model.gate.set()
        self.scheduler.close()

    def jobs(self, count, key="base"):
        return [Job(key, str(i), "Chinese" if i % 2 else "English", prompt=f"voice-{i}", instruct=f"style-{i}") for i in range(count)]

    def run_jobs(self, jobs):
        futures = [self.scheduler.submit(j) for j in jobs]
        self.scheduler.start()
        return futures

    def warm(self, job):
        self.scheduler.policy.succeeded([job] * 8, 4 * GIB)

    def test_native_batch_growth_and_ordered_voice_mapping(self):
        jobs = self.jobs(7)
        futures = self.run_jobs(jobs)
        self.assertEqual([f.result(5)[0] for f in futures], [f"{i}:voice-{i}" for i in range(7)])
        self.assertEqual([len(c[0]) for c in self.model.calls], [1, 2, 4])
        self.assertEqual(self.model.calls[1][1], ["Chinese", "English"])

    def test_mixed_models_permissions_and_lengths_are_not_batched(self):
        jobs = self.jobs(2) + self.jobs(2, "voicedesign") + [Job("base", "长" * 300, "Chinese", prompt="long")]
        jobs[3].allow_download = True
        self.warm(jobs[0])
        futures = self.run_jobs(jobs)
        for future in futures:
            future.result(5)
        self.assertEqual(self.loaded, [("base", False), ("voicedesign", False), ("voicedesign", True), ("base", False)])
        self.assertEqual(futures[2].result()[0], "0:style-0")

    def test_oom_splits_and_releases_exception_temporaries(self):
        jobs = self.jobs(4)
        self.warm(jobs[0])
        self.model.maximum = 2
        futures = self.run_jobs(jobs)
        self.assertTrue(all(f.result(5)[1] == 24000 for f in futures))
        self.assertEqual([len(c[0]) for c in self.model.calls], [4, 2, 2])
        self.assertEqual(self.memory.recovered, 1)
        self.assertEqual(self.scheduler.status()["oom_retries"], 1)

    def test_bad_item_does_not_poison_other_results(self):
        jobs = self.jobs(3)
        jobs[1].text = "bad"
        self.warm(jobs[0])
        futures = self.run_jobs(jobs)
        self.assertEqual(futures[0].result(5)[0], "0:voice-0")
        with self.assertRaisesRegex(InferenceError, "bad prompt"):
            futures[1].result(5)
        self.assertEqual(futures[2].result(5)[0], "2:voice-2")

    def test_single_oom_reports_actionable_error_and_next_job_runs(self):
        jobs = self.jobs(2)
        jobs[0].text = "oom"
        futures = self.run_jobs(jobs)
        with self.assertRaisesRegex(InferenceError, "单条任务"):
            futures[0].result(5)
        self.assertEqual(futures[1].result(5)[0], "1:voice-1")

    def test_pressure_refuses_work_without_exhausting_memory(self):
        self.memory.current = replace(self.memory.current, free=100 * 1024**2)
        futures = self.run_jobs(self.jobs(1))
        with self.assertRaisesRegex(InferenceError, "安全余量"):
            futures[0].result(5)
        self.assertEqual(self.model.calls, [])

    def test_cancelled_pending_job_is_not_generated(self):
        jobs = self.jobs(3)
        futures = [self.scheduler.submit(j) for j in jobs]
        futures[1].cancel()
        self.scheduler.start()
        futures[0].result(5)
        futures[2].result(5)
        self.assertFalse(any("1" in c[0] for c in self.model.calls))

    def test_bounded_queue_and_shutdown_resolve_all_waiters(self):
        self.scheduler.settings = replace(self.settings, max_pending_jobs=2)
        first, second, third = self.jobs(3)
        self.scheduler.submit(first)
        self.scheduler.submit(second)
        with self.assertRaises(InferenceError) as caught:
            self.scheduler.submit(third)
        self.assertEqual(caught.exception.status_code, 429)
        self.scheduler.close()
        for job in (first, second):
            with self.assertRaisesRegex(InferenceError, "关闭"):
                job.future.result(1)

    def test_model_load_failure_does_not_kill_worker(self):
        original = self.scheduler.model_loader

        def load(key, allow):
            if key == "voicedesign":
                raise InferenceError("请先下载模型", 409)
            return original(key, allow)

        self.scheduler.model_loader = load
        jobs = self.jobs(1, "voicedesign") + self.jobs(1)
        futures = self.run_jobs(jobs)
        with self.assertRaisesRegex(InferenceError, "下载"):
            futures[0].result(5)
        self.assertEqual(futures[1].result(5)[0], "0:voice-0")

    def test_worker_does_not_retain_model_during_switch(self):
        owner = [None]

        def loader(key, allow):
            previous = weakref.ref(owner[0]) if owner[0] is not None else None
            owner[0] = None
            if previous is not None:
                self.assertIsNone(previous(), "worker kept old GPU model alive")
            owner[0] = FakeModel()
            return owner[0]

        self.scheduler.model_loader = loader
        futures = self.run_jobs(self.jobs(1) + self.jobs(1, "voicedesign"))
        self.assertEqual(futures[0].result(5)[0], "0:voice-0")
        self.assertEqual(futures[1].result(5)[0], "0:style-0")

    def test_status_stays_responsive_and_shutdown_waits_for_active_call(self):
        self.model.gate = threading.Event()
        jobs = self.jobs(2)
        jobs[0].clip_id = "active-clip"
        futures = self.run_jobs(jobs)
        self.assertTrue(self.model.entered.wait(2))
        self.assertEqual(self.scheduler.status()["active_clip_ids"], ["active-clip"])
        stopped = threading.Event()
        closer = threading.Thread(target=lambda: (self.scheduler.close(), stopped.set()))
        closer.start()
        self.assertFalse(stopped.wait(0.05))
        self.model.gate.set()
        self.assertTrue(stopped.wait(2))
        closer.join()
        futures[0].result(1)
        with self.assertRaises(InferenceError):
            futures[1].result(1)

    def test_batch_timing_is_shared_and_measures_generation_not_audio_duration(self):
        jobs = self.jobs(2)
        self.warm(jobs[0])
        self.model.gate = threading.Event()
        clock = [1000.0]
        with patch('qwen3_tts_web.inference.time', SimpleNamespace(monotonic=lambda: clock[0])):
            futures = self.run_jobs(jobs)
            self.assertTrue(self.model.entered.wait(2))
            state = self.scheduler.status()
            self.assertEqual(state["active"], 2)
            self.assertEqual(state["active_phase"], "generating")
            clock[0] += 2.5
            self.model.gate.set()
            for future in futures:
                future.result(5)
        self.assertEqual(jobs[0].generation_stats, jobs[1].generation_stats)
        self.assertEqual(jobs[0].generation_stats["batch_size"], 2)
        self.assertEqual(jobs[0].generation_stats["batch_seconds"], 2.5)

    def test_failed_generation_does_not_mark_length_class_calibrated(self):
        job = Job('base', 'oom', 'Chinese', prompt='voice')
        future = self.run_jobs([job])[0]
        with self.assertRaises(InferenceError):
            future.result(5)
        self.assertEqual(self.scheduler.policy.profile(job).proven_batch, 0)
        self.assertEqual(self.scheduler.status()['completed'], 0)

    def test_cold_calibration_has_explicit_phase(self):
        self.model.gate = threading.Event()
        jobs = self.jobs(2)
        jobs[0].clip_id, jobs[1].clip_id = "one", "two"
        futures = self.run_jobs(jobs)
        self.assertTrue(self.model.entered.wait(2))
        state = self.scheduler.status()
        self.assertEqual(state["active_phase"], "calibrating")
        self.assertEqual(state["active_clip_ids"], ["one"])
        self.assertEqual(state["queued_clip_ids"], ["two"])
        self.model.gate.set()
        for future in futures:
            future.result(5)

    def test_cache_is_released_only_after_cuda_queue_is_idle(self):
        self.scheduler.idle_cache_seconds = 0.05
        self.memory.current = replace(self.memory.current, reserved=8 * GIB)
        self.model.gate = threading.Event()
        futures = self.run_jobs(self.jobs(2))
        self.assertTrue(self.model.entered.wait(2))
        self.assertFalse(self.memory.idle_released.wait(0.1))
        self.model.gate.set()
        for future in futures:
            future.result(5)
        self.assertTrue(self.memory.idle_released.wait(2))
        self.assertEqual(self.scheduler.status()["memory"]["cached_gb"], 0)
        self.assertEqual(self.scheduler.status()["memory"]["allocated_gb"], 4)

    def test_mps_does_not_run_new_idle_cache_cleanup(self):
        self.memory.cuda = False
        self.scheduler.idle_cache_seconds = 0.01
        for future in self.run_jobs(self.jobs(1)):
            future.result(5)
        self.assertFalse(self.memory.idle_released.wait(0.08))

    def test_idle_cleanup_failure_does_not_kill_worker(self):
        self.scheduler.idle_cache_seconds = 0.01
        with patch.object(self.memory, "release_idle_cache", side_effect=RuntimeError("cleanup failed")) as cleanup:
            self.run_jobs(self.jobs(1))[0].result(5)
            deadline = time.monotonic() + 2
            while not cleanup.called and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(cleanup.called)
        self.scheduler.submit(Job("base", "next", "Chinese", prompt="voice")).result(5)


if __name__ == "__main__":
    unittest.main()
