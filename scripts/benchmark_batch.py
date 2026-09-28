"""Real-device comparison. Stop the service first; no catalog data is changed.

Example: .venv/Scripts/python scripts/benchmark_batch.py --prompt pt/role/voice.pt
Only load trusted prompt files. Measurements use bounded, short test utterances.
"""

import argparse
import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from qwen3_tts_web.config import load_settings
from qwen3_tts_web.inference import InferenceScheduler, Job
from qwen3_tts_web.runtime import load


class BoundedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.batch_sizes = []
        self.peaks = []

    def generate_voice_clone(self, **kwargs):
        self.batch_sizes.append(len(kwargs["text"]))
        result = self.model.generate_voice_clone(**kwargs, max_new_tokens=256)
        if str(self.device).startswith("cuda"):
            self.peaks.append(torch.cuda.max_memory_allocated(self.device))
        return result


def run_case(settings, target, prompt, count, label, scheduler=None):
    scheduler = scheduler or InferenceScheduler(settings, lambda *_: target, threading.Lock())
    if scheduler.worker.ident is None:
        scheduler.start()
    target.batch_sizes.clear()
    target.peaks.clear()
    jobs = [Job(settings.model_key, f"这是第{i + 1}条语音测试，我们正在检查并行生成的速度和稳定性。",
                "Chinese", prompt=prompt) for i in range(count)]
    start = time.perf_counter()
    futures = [scheduler.submit(job) for job in jobs]
    audio_seconds = 0
    for future in futures:
        wav, sample_rate = future.result(timeout=900)
        wav = np.asarray(wav)
        assert wav.size and np.isfinite(wav).all() and np.max(np.abs(wav)) > 1e-5
        audio_seconds += wav.size / sample_rate
    elapsed = time.perf_counter() - start
    report = {"case": label, "clips": count, "seconds": elapsed,
              "audio_seconds": audio_seconds, "clips_per_second": count / elapsed,
              "audio_seconds_per_second": audio_seconds / elapsed,
              "batch_sizes": list(target.batch_sizes),
              "peak_allocated_gb": max(target.peaks, default=0) / 1024**3,
              "scheduler": scheduler.status()}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return scheduler, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--extra-counts", nargs="*", type=int, default=[])
    parser.add_argument("--output", type=Path, default=Path(".logs/batch-benchmark.json"))
    args = parser.parse_args()
    settings = replace(load_settings(), offline=True)
    with FileLock(str(settings.root / ".runtime.lock"), timeout=0):
        target = BoundedModel(load(settings))
        prompts = torch.load(args.prompt, map_location="cpu", weights_only=False)
        prompt = prompts[0] if isinstance(prompts, (list, tuple)) else prompts
        # Warm up CUDA / decoder kernels equally before either timed case.
        target.generate_voice_clone(text=["你好，这是预热。"], language=["Chinese"], voice_clone_prompt=[prompt])
        reports = []
        if not args.skip_baseline:
            serial, report = run_case(replace(settings, max_batch_size=1), target, prompt, args.count, "serial")
            reports.append(report)
            serial.close()
        adaptive, report = run_case(settings, target, prompt, args.count, "adaptive_cold")
        reports.append(report)
        adaptive, report = run_case(settings, target, prompt, args.count, "adaptive_warm", adaptive)
        reports.append(report)
        for count in args.extra_counts:
            adaptive, report = run_case(settings, target, prompt, count, f"adaptive_{count}", adaptive)
            reports.append(report)
        adaptive.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"device": str(target.device), "torch": torch.__version__,
                                           "max_new_tokens": 256, "reports": reports},
                                          ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
