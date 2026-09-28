# HTTP API; model initialization belongs to the application lifespan.
"""
Qwen3-TTS Voice Clone API（角色/情感分级版）

pt 目录结构：
    pt/
      张琛/
        张琛_平静.pt
        张琛_愤怒.pt
      李四/
        李四_激动.pt
"""

import dataclasses
import asyncio
import inspect
import json
import os
import re
import tempfile
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from .config import load_settings
from .library import AudioLibrary
from .runtime import apply_memory_limit, free, load
from .inference import InferenceError, InferenceScheduler, Job

settings = load_settings()
BASE_DIR = settings.root


# ================= 目录 =================
PT_DIR = BASE_DIR / "pt"
OUT_DIR = BASE_DIR / "output"
WEB_DIR = Path(__file__).parent / "web"


# ================= 定位 VoiceClonePromptItem =================
def _find_prompt_item_cls():
    for mod_name in (
        "qwen_tts.inference.qwen3_tts_model",
        "qwen_tts.inference",
        "qwen_tts",
    ):
        try:
            mod = __import__(mod_name, fromlist=["VoiceClonePromptItem"])
            cls = getattr(mod, "VoiceClonePromptItem", None)
            if cls is not None:
                return cls, mod_name
        except Exception:
            continue
    return None, None


VOICE_CLONE_PROMPT_ITEM = None
model = None
active_model_key = settings.model_key
inference_lock = threading.Lock()
generation_claims = set()


def get_model(key, allow_download=False):
    """Caller holds inference_lock; keep only one inference model on the device."""
    global model, active_model_key
    from .models import model_path, prepare_model, validate_model

    if model is not None and active_model_key == key:
        return model
    errors = validate_model(model_path(settings, key), key)
    if errors and (settings.offline or not allow_download):
        raise HTTPException(
            409,
            detail={
                "message": "模型缺失或不完整；离线模式禁止下载"
                if settings.offline
                else "请先下载模型或确认本次下载",
                "model_key": key,
                "missing": errors,
            },
        )
    # Prepare first: a failed download must not evict a working model.
    prepare_model(settings, key=key)
    app.state.ready = False
    model = None
    free()
    model = load(settings, key=key)
    active_model_key = key
    app.state.ready = True
    return model


@asynccontextmanager
async def lifespan(app):
    global model, VOICE_CLONE_PROMPT_ITEM, active_model_key
    for directory in (PT_DIR, OUT_DIR, UPLOAD_DIR, BASE_DIR / "references"):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        model = load(settings)
        active_model_key = settings.model_key
        AudioLibrary(OUT_DIR).list()
        VOICE_CLONE_PROMPT_ITEM, _ = _find_prompt_item_cls()
        app.state.scheduler = InferenceScheduler(settings, get_model, inference_lock)
        app.state.scheduler.start()
        app.state.ready = True
        yield
    finally:
        app.state.ready = False
        if app.state.scheduler is not None:
            await run_in_threadpool(app.state.scheduler.close)
            app.state.scheduler = None
        model = None
        free()


# ================= FastAPI =================
app = FastAPI(
    title="Qwen3-TTS Voice Clone API",
    description="按 角色/情感 组织 pt 文件，生成克隆语音",
    version="3.0.0",
    lifespan=lifespan,
)
app.state.ready = False
app.state.scheduler = None
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/static/audio/{filename}", include_in_schema=False)
def audio_file(filename: str):
    try:
        path = AudioLibrary(OUT_DIR).audio_path(filename)
    except ValueError:
        raise HTTPException(404, detail="音频不存在")
    if not path.is_file():
        raise HTTPException(404, detail="音频不存在")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/health", summary="模型就绪状态")
def api_health():
    from fastapi.responses import JSONResponse

    from .models import MODELS

    ready = app.state.ready and model is not None
    return JSONResponse(
        {
            "ready": ready,
            "device": str(getattr(model, "device", settings.device)),
            "model": MODELS[active_model_key],
        },
        status_code=200 if ready else 503,
    )


# ================= 请求模型 =================
def normalize_emotion(value):
    if value is None:
        return "平静"
    if isinstance(value, str):
        return value.strip() or "平静"
    return value


Emotion = Annotated[str, BeforeValidator(normalize_emotion)]


class TTSRequest(BaseModel):
    role: str = ""
    emotion: Emotion = "平静"
    text: str = Field(min_length=1, max_length=10000)
    language: str = "Chinese"
    mode: str = "url"  # "url" | "file"
    pt_file: str = ""  # 兼容旧接口
    clip_id: str | None = None


class VoiceDesignRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    instruct: str = Field(min_length=1, max_length=2000)
    language: str = "Chinese"
    mode: Literal["url", "file"] = "url"
    clip_id: str | None = None
    allow_download: bool = False

    @field_validator("text", "instruct")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("文本与音色描述不能为空")
        return value.strip()


class BatchCloneRequest(TTSRequest):
    synthesis_mode: Literal["clone"] = "clone"
    mode: Literal["url"] = "url"


class BatchDesignRequest(VoiceDesignRequest):
    synthesis_mode: Literal["design"] = "design"
    mode: Literal["url"] = "url"


class BatchRequest(BaseModel):
    items: list[Annotated[BatchCloneRequest | BatchDesignRequest,
                          Field(discriminator="synthesis_mode")]] = Field(min_length=1, max_length=32)


class ClipEdit(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    title: str = Field(default="", max_length=200)
    text: str = Field(default="", max_length=10000)
    role: str = Field(default="", max_length=200)
    emotion: Emotion = "平静"
    language: str = Field(default="Chinese", max_length=40)
    synthesis_mode: Literal["clone", "design", "legacy"] = "clone"
    instruct: str = Field(default="", max_length=2000)
    speaker: Literal["raw", "clean", "radio", "dirty", "broken", "destroyed"] = "raw"
    signal: Literal["raw", "strong", "normal", "weak", "critical"] = "raw"
    delay: float = Field(default=0, ge=0, le=3600)
    position: float = Field(default=0, ge=0)


class ExportRequest(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=1000)


def library_record(clip_id):
    try:
        return AudioLibrary(OUT_DIR).get(clip_id)
    except KeyError:
        raise HTTPException(404, detail="音频记录不存在")


@app.get("/api/clips")
def list_clips():
    return {"clips": AudioLibrary(OUT_DIR).list()}


@app.post("/api/clips", status_code=201)
def create_clip(req: ClipEdit):
    return AudioLibrary(OUT_DIR).create(req.model_dump())


@app.get("/api/clips/{clip_id}")
def read_clip(clip_id: str):
    return library_record(clip_id)


@app.patch("/api/clips/{clip_id}")
def update_clip(clip_id: str, req: ClipEdit):
    try:
        return AudioLibrary(OUT_DIR).update(clip_id, req.model_dump(exclude_unset=True))
    except KeyError:
        raise HTTPException(404, detail="音频记录不存在")


@app.delete("/api/clips/{clip_id}")
def delete_clip(clip_id: str):
    try:
        AudioLibrary(OUT_DIR).delete(clip_id)
    except KeyError:
        raise HTTPException(404, detail="音频记录不存在")
    return {"status": "ok"}


def export_name(record):
    title = re.sub(
        r'[\x00-\x1f<>:"/\\|?*]',
        "_",
        record.get("title") or record.get("text") or "audio",
    )
    return f"{title[:80].strip('. ') or 'audio'}-{record['id'][:8]}.wav"


@app.get("/api/clips/{clip_id}/download")
def download_clip(clip_id: str):
    record = library_record(clip_id)
    if not record["available"]:
        raise HTTPException(404, detail="音频文件不存在")
    return FileResponse(
        AudioLibrary(OUT_DIR).audio_path(record["filename"]),
        media_type="audio/wav",
        filename=export_name(record),
    )


@app.post("/api/clips/export")
def export_clips(req: ExportRequest):
    library = AudioLibrary(OUT_DIR)
    with library.lock:
        records = [library_record(key) for key in dict.fromkeys(req.ids)]
        if any(not record["available"] for record in records):
            raise HTTPException(409, detail="所选记录包含缺失的音频，请刷新列表")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as handle:
            target = Path(handle.name)
        try:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for index, record in enumerate(records, 1):
                    archive.write(
                        library.audio_path(record["filename"]),
                        f"{index:03d}-{export_name(record)}",
                    )
                archive.writestr(
                    "manifest.json", json.dumps(records, ensure_ascii=False, indent=2)
                )
        except BaseException:
            target.unlink(missing_ok=True)
            raise
    return FileResponse(
        target,
        filename="qwen3-tts-audio.zip",
        media_type="application/zip",
        background=BackgroundTask(target.unlink, missing_ok=True),
    )


@app.get("/api/capabilities")
def capabilities():
    from .models import MODELS, model_path, validate_model

    errors = validate_model(model_path(settings, "voicedesign"), "voicedesign")
    cuda = str(getattr(model, "device", settings.device)).startswith("cuda")
    return {
        "voice_design": {
            "model": MODELS["voicedesign"],
            "installed": not errors,
            "offline": settings.offline,
        },
        "active_model": MODELS[active_model_key],
        "inference": {
            "mode": "adaptive_batch" if cuda else "serial",
            "max_batch_size": settings.max_batch_size if cuda else 1,
            "max_request_items": min(32, settings.max_pending_jobs) if cuda else 1,
            "gpu_memory_fraction": settings.gpu_memory_fraction,
        },
    }


@app.get("/api/inference/status", summary="自适应生成队列与最近一次显存测量")
async def inference_status():
    scheduler = app.state.scheduler
    if scheduler is None:
        raise HTTPException(503, detail="推理队列尚未就绪")
    return scheduler.status()


class PromptCreateRequest(BaseModel):
    ref_audio: str
    ref_text: str = ""
    role: str
    emotion: Emotion = "平静"
    overwrite: bool = False


# ================= pt 扫描与定位 =================
def list_pt_roles() -> Dict[str, List[str]]:
    """扫描 pt/{角色}/{角色}_{情感}.pt，返回 {角色: [情感, ...]}（有序）"""
    result: Dict[str, List[str]] = {}
    if not PT_DIR.exists():
        return result

    for role_dir in sorted(PT_DIR.iterdir(), key=lambda p: p.name):
        if not role_dir.is_dir():
            continue
        role = role_dir.name
        prefix = role + "_"
        emotions: List[str] = []
        for f in sorted(role_dir.glob("*.pt"), key=lambda p: p.name):
            stem = f.stem
            emo = stem[len(prefix) :] if stem.startswith(prefix) else stem
            if emo not in emotions:
                emotions.append(emo)
        if emotions:
            result[role] = emotions
    return result


def resolve_pt_path(role: str, emotion: str):
    """根据 角色+情感 定位 .pt 文件；找不到返回 None。"""
    if not role or not emotion:
        return None
    role_dir = PT_DIR / role
    if not role_dir.is_dir():
        return None

    # ① 标准命名 {角色}_{情感}.pt
    p = role_dir / f"{role}_{emotion}.pt"
    if p.is_file():
        return p

    # ② 备用命名 {情感}.pt
    p = role_dir / f"{emotion}.pt"
    if p.is_file():
        return p

    # ③ 扫描匹配（兼容历史命名）
    prefix = role + "_"
    for f in role_dir.glob("*.pt"):
        stem = f.stem
        emo_name = stem[len(prefix) :] if stem.startswith(prefix) else stem
        if emo_name == emotion:
            return f
    return None


def flat_pt_names() -> List[str]:
    """扁平列出所有 '角色/情感'，用于错误提示。"""
    out = []
    for role, emos in list_pt_roles().items():
        for e in emos:
            out.append(f"{role}/{e}")
    return out


# ================= 工具函数 =================
def _dict_to_item(d: dict):
    cls = VOICE_CLONE_PROMPT_ITEM
    if cls is None:
        return SimpleNamespace(**d)

    if dataclasses.is_dataclass(cls):
        valid = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in valid}
        try:
            return cls(**kw)
        except TypeError:
            for name in valid - kw.keys():
                kw[name] = None
            return cls(**kw)

    try:
        sig = inspect.signature(cls)
        valid = set(sig.parameters.keys())
        kw = {k: v for k, v in d.items() if k in valid}
        return cls(**kw)
    except Exception as e:
        print(f"[warn] 构造 {cls.__name__} 失败: {e}，退化到 SimpleNamespace")
        return SimpleNamespace(**d)


def _normalize_item(obj: Any):
    if isinstance(obj, dict):
        return _dict_to_item(obj)
    return obj


def _unwrap_prompt(obj: Any) -> List[Any]:
    if obj is None:
        raise ValueError("pt 文件内容为空")

    if (
        isinstance(obj, dict)
        and "items" in obj
        and isinstance(obj["items"], (list, tuple))
    ):
        print(f"[unwrap] 检测到 'items' 包装，解包 {len(obj['items'])} 个元素")
        obj = list(obj["items"])

    if isinstance(obj, (list, tuple)):
        items = list(obj)
    else:
        items = [obj]

    if not items:
        raise ValueError("pt 文件内容为空")
    return [_normalize_item(it) for it in items]


def load_prompt_from_path(p: Path) -> List[Any]:
    import torch

    if p is None or not p.is_file():
        raise HTTPException(404, detail=f"pt 文件不存在: {p}")

    try:
        obj = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:
        raise HTTPException(500, detail=f"加载 pt 失败: {e}")

    try:
        prompt_list = _unwrap_prompt(obj)
    except Exception as e:
        raise HTTPException(500, detail=f"pt 内容非法: {e}")

    it0 = prompt_list[0]
    has_emb = getattr(it0, "ref_spk_embedding", None) is not None
    has_code = getattr(it0, "ref_code", None) is not None
    xonly = getattr(it0, "x_vector_only_mode", None)
    rel = p.relative_to(PT_DIR) if PT_DIR in p.parents else p.name

    print(
        f"[load_prompt] {rel} -> {len(prompt_list)} item(s), type={type(it0).__name__}"
    )
    print(
        f"[load_prompt] ref_spk_embedding={'yes' if has_emb else 'no'}, "
        f"ref_code={'yes' if has_code else 'no'}, x_vector_only_mode={xonly}"
    )
    return prompt_list


def _state_for_debug(prompt_list):
    it0 = prompt_list[0]
    if dataclasses.is_dataclass(it0):
        return {
            f.name: type(getattr(it0, f.name)).__name__ for f in dataclasses.fields(it0)
        }
    if isinstance(it0, SimpleNamespace):
        return {k: type(v).__name__ for k, v in vars(it0).items()}
    return {
        a: type(getattr(it0, a)).__name__ for a in dir(it0) if not a.startswith("_")
    }


# ================= API：pt 管理 =================
@app.get("/api/pt/list", summary="按角色列出所有 pt 音色")
def api_pt_list():
    roles = list_pt_roles()
    total = sum(len(v) for v in roles.values())
    return {
        "roles": roles,
        "count": total,
        "role_count": len(roles),
        "dir": str(PT_DIR),
    }


_SAFE = re.compile(r"^[\w\u4e00-\u9fa5\-]{1,30}$")


@app.post("/api/pt/create", summary="从参考音频创建 pt（保存到 角色/情感 目录）")
def api_pt_create(req: PromptCreateRequest):
    import torch

    if not os.path.exists(req.ref_audio):
        raise HTTPException(404, detail=f"参考音频不存在: {req.ref_audio}")
    if not req.role:
        raise HTTPException(400, detail="必须提供 role")
    if not _SAFE.match(req.role) or not _SAFE.match(req.emotion):
        raise HTTPException(
            400, detail="role/emotion 只能包含字母/数字/中文/下划线/横线"
        )
    role_dir = PT_DIR / req.role
    role_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{req.role}_{req.emotion}.pt"
    path = role_dir / fname

    if path.exists() and not req.overwrite:
        raise HTTPException(
            400, detail=f"文件已存在: {req.role}/{fname}（可传 overwrite=true 覆盖）"
        )

    kwargs = {"ref_audio": req.ref_audio}
    if req.ref_text:
        kwargs["ref_text"] = req.ref_text
    else:
        kwargs["x_vector_only_mode"] = True
        print("[create] 未提供 ref_text，使用 x_vector_only 模式")

    try:
        with inference_lock:
            target = get_model(settings.model_key)
            apply_memory_limit(str(target.device), settings)
            prompt = target.create_voice_clone_prompt(**kwargs)
            del target
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(500, detail=f"创建 prompt 失败: {e}")

    prompt_list = _unwrap_prompt(prompt)
    print(f"[create] {len(prompt_list)} item(s), type={type(prompt_list[0]).__name__}")
    print(f"[create] 字段: {_state_for_debug(prompt_list)}")

    try:
        torch.save(prompt_list, str(path))
    except Exception as e:
        raise HTTPException(500, detail=f"保存 pt 失败: {e}")

    return {
        "status": "ok",
        "role": req.role,
        "emotion": req.emotion,
        "rel": f"{req.role}/{fname}",
        "path": str(path),
        "items": len(prompt_list),
        "item_type": type(prompt_list[0]).__name__,
        "x_vector_only": not bool(req.ref_text),
    }


@app.get("/api/pt/inspect", summary="诊断 pt 结构")
def api_pt_inspect(role: str = "", emotion: str = "平静", name: str = ""):
    import torch

    emotion = normalize_emotion(emotion)
    if role:
        p = resolve_pt_path(role, emotion)
        if p is None:
            raise HTTPException(404, detail=f"未找到: {role}/{emotion}")
    elif name:
        p = PT_DIR / name
        if not p.is_file():
            raise HTTPException(404, detail=f"pt 不存在: {name}")
    else:
        raise HTTPException(400, detail="需要 role 或 name")

    obj = torch.load(str(p), map_location="cpu", weights_only=False)

    def _describe(o, depth=0, max_depth=2):
        if depth > max_depth:
            return type(o).__name__
        if isinstance(o, dict):
            return {
                k: _describe(v, depth + 1, max_depth) for k, v in list(o.items())[:10]
            }
        if isinstance(o, (list, tuple)):
            return [_describe(o[0], depth + 1, max_depth)] if o else []
        if isinstance(o, torch.Tensor):
            return f"Tensor{tuple(o.shape)}"
        try:
            return {
                f.name: _describe(getattr(o, f.name), depth + 1, max_depth)
                for f in dataclasses.fields(o)
            }
        except Exception:
            pass
        return f"{type(o).__name__}"

    return {
        "role": role or None,
        "emotion": emotion or None,
        "path": str(p.relative_to(PT_DIR)),
        "structure": _describe(obj),
    }


# ================= API：TTS =================
@app.post("/api/tts", summary="生成语音（role+emotion 或 pt_file）")
async def api_tts(req: TTSRequest):
    async with claim_generation(req.clip_id):
        prompt_list = await run_in_threadpool(prepare_tts, req)
        job = Job(
            key=settings.model_key, text=req.text, language=req.language,
            prompt=prompt_list[0], clip_id=req.clip_id,
        )
        wav, sr = await schedule(job)
        generated = {
            "role": req.role, "emotion": req.emotion, "text": req.text,
            "language": req.language, "synthesis_mode": "clone", "instruct": "",
        }
        return await run_in_threadpool(save_generation, req, [wav], sr, generated, job.generation_stats)


@asynccontextmanager
async def claim_generation(clip_id):
    # All callers run on the event loop. Hold the claim through disk persistence,
    # so two tabs cannot race to replace the same clip after inference finishes.
    if clip_id and clip_id in generation_claims:
        raise HTTPException(409, detail="这条语音正在生成，请等待完成")
    if clip_id:
        generation_claims.add(clip_id)
    try:
        yield
    finally:
        generation_claims.discard(clip_id)


async def schedule(job):
    scheduler = app.state.scheduler
    if scheduler is None:
        raise HTTPException(503, detail="推理队列尚未就绪")
    try:
        return await asyncio.wrap_future(scheduler.submit(job))
    except InferenceError as exc:
        raise HTTPException(exc.status_code, detail=str(exc)) from None


def prepare_tts(req):
    if not req.text.strip():
        raise HTTPException(400, detail="文本不能为空")
    if req.clip_id:
        library_record(req.clip_id)
    if req.mode not in ("file", "url"):
        raise HTTPException(400, detail="mode 必须是 'file' 或 'url'")
    # 定位 .pt
    if req.pt_file:
        p = PT_DIR / req.pt_file
    else:
        if not req.role:
            raise HTTPException(400, detail="必须提供 role")
        p = resolve_pt_path(req.role, req.emotion)
        if p is None:
            raise HTTPException(
                404,
                detail=f"未找到音色: {req.role}/{req.emotion}；可用: {flat_pt_names()[:20]}",
            )

    prompt_list = load_prompt_from_path(p)
    if len(prompt_list) != 1:
        raise HTTPException(400, detail="单条语音需要一个音色 prompt；请使用只包含一个音色的 .pt 文件")
    return prompt_list


def save_generation(req, wavs, sr, generated, generation_stats=None):
    metadata = {"generated": generated, "generation_stats": generation_stats or {}}
    if not req.clip_id:
        metadata.update(generated, title=req.text[:60])
    try:
        record = AudioLibrary(OUT_DIR).save_audio(wavs[0], sr, metadata, req.clip_id)
    except KeyError:
        raise HTTPException(404, detail="记录已被删除，本次生成结果未保存")
    except Exception as e:
        raise HTTPException(500, detail=f"保存音频失败: {e}")
    fname = record["filename"]
    out_path = OUT_DIR / fname
    if req.mode == "file":
        return FileResponse(str(out_path), media_type="audio/wav", filename=fname)
    return {
        "status": "ok",
        "filename": fname,
        "url": record["url"],
        "local_path": str(out_path),
        "role": generated.get("role", ""),
        "emotion": generated.get("emotion", "平静"),
        "clip": record,
    }


@app.post("/api/voice-design", summary="通过自然语言描述设计音色并生成语音")
async def api_voice_design(req: VoiceDesignRequest):
    async with claim_generation(req.clip_id):
        if req.clip_id:
            await run_in_threadpool(library_record, req.clip_id)
        job = Job(
            key="voicedesign", text=req.text, language=req.language,
            instruct=req.instruct, allow_download=req.allow_download, clip_id=req.clip_id,
        )
        wav, sr = await schedule(job)
        generated = {
            "text": req.text, "language": req.language, "instruct": req.instruct,
            "synthesis_mode": "design", "role": "", "emotion": "平静",
        }
        return await run_in_threadpool(save_generation, req, [wav], sr, generated, job.generation_stats)


@app.post("/api/generate-batch", summary="批量提交语音，每条任务独立返回结果")
async def api_generate_batch(req: BatchRequest):
    async def generate(item):
        try:
            result = await (api_tts(item) if item.synthesis_mode == "clone" else api_voice_design(item))
            return {"status": "ok", "result": result}
        except HTTPException as exc:
            return {"status": "error", "status_code": exc.status_code, "detail": exc.detail}
        except Exception:
            import logging

            logging.getLogger(__name__).exception("生成任务失败")
            return {"status": "error", "status_code": 500, "detail": "生成失败，请查看服务日志"}

    # Waiting on futures consumes no worker-pool thread, leaving health, uploads
    # and the library responsive even with a large inference backlog.
    return {"items": await asyncio.gather(*(generate(item) for item in req.items))}


# ================= 上传 & maker 页面 =================
UPLOAD_DIR = BASE_DIR / "uploads"

ALLOWED_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".aac"}


@app.post("/api/pt/upload", summary="上传参考音频，返回服务器路径")
async def api_pt_upload(file: UploadFile = File(...)):
    name = file.filename or "upload.wav"
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            400,
            detail=f"不支持的音频格式: {ext}（支持 {sorted(ALLOWED_EXT)}）",
        )

    safe = f"{uuid.uuid4().hex}{ext}"
    dst = UPLOAD_DIR / safe
    content = await file.read()
    dst.write_bytes(content)

    print(f"[upload] {name} -> {dst} ({len(content)} bytes)")
    return {
        "status": "ok",
        "path": str(dst),
        "filename": safe,
        "original_name": name,
        "size": len(content),
    }


@app.get("/api/pt/download", summary="下载 pt 文件")
def api_pt_download(role: str, emotion: str = "平静"):
    emotion = normalize_emotion(emotion)
    p = resolve_pt_path(role, emotion)
    if p is None:
        raise HTTPException(404, detail=f"未找到: {role}/{emotion}")
    return FileResponse(
        str(p),
        media_type="application/octet-stream",
        filename=p.name,
    )


@app.get("/maker", response_class=HTMLResponse, include_in_schema=False)
def maker_page():
    p = WEB_DIR / "maker.html"
    if not p.exists():
        return HTMLResponse(
            "<h3>web/maker.html 不存在</h3>"
            "<p>请把 maker.html 放到 <code>web/</code> 目录。</p>"
        )
    return HTMLResponse(p.read_text(encoding="utf-8"))


# ================= 测试页面 =================
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    idx = WEB_DIR / "index.html"
    if not idx.exists():
        return HTMLResponse(
            "<h3>web/index.html 不存在</h3>"
            "<p>请把测试页面放到 <code>web/index.html</code>，或访问 "
            "<a href='/docs'>/docs</a>。</p>"
        )
    return HTMLResponse(idx.read_text(encoding="utf-8"))


# ================= 启动 =================
if __name__ == "__main__":
    from .cli import main

    raise SystemExit(main(["serve"]))
