"""Shared configuration, with no imports of the inference stack."""

import os
import re
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import tomllib


def project_root(root=None):
    if root or os.environ.get("QWEN3_TTS_ROOT"):
        return Path(root or os.environ["QWEN3_TTS_ROOT"]).expanduser().resolve()
    source = Path(__file__).resolve().parents[2]
    if (source / "pyproject.toml").is_file() and (source / "scripts").is_dir():
        return source
    return Path.cwd().resolve()


@dataclass(frozen=True)
class Settings:
    root: Path
    device: str = "auto"
    model: str = "1.7B"
    source: str = "domestic"
    host: str = "0.0.0.0"
    port: int = 8001
    offline: bool = False
    pip_index: str = ""
    torch_index: str = ""
    hf_endpoint: str = ""
    max_batch_size: int = 64
    gpu_memory_fraction: float = 0.90
    gpu_memory_reserve_gb: float = 1.0
    batch_wait_ms: int = 80
    max_pending_jobs: int = 128

    def __post_init__(self):
        if not re.fullmatch(r"auto|mps|cuda(?::[0-9]+)?", self.device):
            raise ValueError("device 必须是 auto、mps 或 cuda:N（本版本不支持纯 CPU）")
        if self.model not in ("1.7B", "0.6B"):
            raise ValueError("model 必须是 1.7B 或 0.6B")
        if self.source not in ("domestic", "official"):
            raise ValueError("source 必须是 domestic 或 official")
        if (
            not isinstance(self.port, int)
            or isinstance(self.port, bool)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port 必须在 1..65535 之间")
        if not isinstance(self.offline, bool):
            raise ValueError("offline 必须是布尔值")
        if not isinstance(self.host, str) or not self.host.strip():
            raise ValueError("host 不能为空")
        for name, low, high in (
            ("max_batch_size", 1, 64),
            ("batch_wait_ms", 0, 1000),
            ("max_pending_jobs", 1, 1024),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} 必须是 {low}..{high} 的整数")
        for name, low, high in (
            ("gpu_memory_fraction", 0.5, 0.95),
            ("gpu_memory_reserve_gb", 0.25, 64),
        ):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} 必须在 {low}..{high} 之间")
        for name in ("pip_index", "torch_index", "hf_endpoint"):
            value = getattr(self, name)
            if not isinstance(value, str) or (
                value and not value.startswith("https://")
            ):
                raise ValueError(f"{name} 必须是 HTTPS 地址或空字符串")

    @property
    def model_key(self):
        return "base" if self.model == "1.7B" else "base-small"

    @property
    def runtime_python(self):
        return (
            self.root
            / ".venv"
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )

    @property
    def url(self):
        host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def environment(self):
        env = os.environ.copy()
        for key, value in asdict(self).items():
            env[f"QWEN3_TTS_{key.upper()}"] = (
                str(value).lower() if isinstance(value, bool) else str(value)
            )
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        return env


def load_settings(root=None, overrides=None):
    root = project_root(root)
    path = root / "config.toml"
    values = {}
    if path.is_file():
        with path.open("rb") as handle:
            values.update(tomllib.load(handle).get("qwen3_tts", {}))
    allowed = {f.name for f in fields(Settings)} - {"root"}
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"未知配置项: {', '.join(sorted(unknown))}")
    for key in allowed:
        name = f"QWEN3_TTS_{key.upper()}"
        if name in os.environ:
            value = os.environ[name]
            if key in {"port", "max_batch_size", "batch_wait_ms", "max_pending_jobs"}:
                value = int(value)
            elif key in {"gpu_memory_fraction", "gpu_memory_reserve_gb"}:
                value = float(value)
            elif key == "offline":
                if value.lower() not in ("true", "false", "1", "0"):
                    raise ValueError(f"{name} 必须是 true/false/1/0")
                value = value.lower() in ("true", "1")
            values[key] = value
    values.update(
        {k: v for k, v in (overrides or {}).items() if k in allowed and v is not None}
    )
    return Settings(root=root, **values)


def save_settings(settings):
    import tomli_w

    path = settings.root / "config.toml"
    document = {}
    if path.exists():
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    values = asdict(settings)
    values.pop("root")
    document["qwen3_tts"] = values
    temporary = path.with_suffix(".toml.tmp")
    temporary.write_text(tomli_w.dumps(document), encoding="utf-8")
    temporary.replace(path)
