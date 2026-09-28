"""Device policy shared by preflight and inference."""

import gc
import os


def select_device(requested="auto", torch_module=None):
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda:0"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            raise RuntimeError(
                "没有可用的 CUDA/MPS 设备；不自动回退 CPU。请运行 doctor。"
            )
    if requested.startswith("cuda"):
        index = int(requested.partition(":")[2] or 0)
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA 设备不可用: {requested}")
        return f"cuda:{index}"
    if requested != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError(f"设备不可用: {requested}")
    return "mps"


def pick_dtype(device, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if device == "mps":
        return torch.float32
    with torch.cuda.device(device):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load(settings, key=None):
    from .models import prepare_model

    device = select_device(settings.device)
    apply_memory_limit(device, settings)
    path = prepare_model(settings, key=key)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from qwen_tts import Qwen3TTSModel

    dtype = pick_dtype(device)
    print(f"[load] {path.name} device={device} dtype={dtype} attn=sdpa", flush=True)
    return Qwen3TTSModel.from_pretrained(
        str(path),
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    )


def apply_memory_limit(device, settings, torch_module=None):
    if torch_module is None:
        import torch as torch_module
    torch = torch_module
    if device.startswith("cuda"):
        # Bound this process's allocator as well as checking live free memory in
        # the scheduler. Other applications and CUDA itself also need headroom.
        torch.cuda.set_per_process_memory_fraction(
            float(settings.gpu_memory_fraction), device
        )


def free():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
