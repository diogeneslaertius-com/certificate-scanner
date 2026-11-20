import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import select

import extract
from db import RECORD_FIELDS, UPLOAD_DIR, Document, Record, Session, init_db

cfg = extract.load_config()
init_db()

app = FastAPI(title="certscan")
app.add_middleware(
    CORSMiddleware, allow_origins=cfg["cors_origins"], allow_methods=["*"], allow_headers=["*"]
)

_calls = {"n": 0}
_lock = threading.Lock()


def get_db():
    with Session() as s:
        yield s


def serialize(rec: Record, doc: Document, cached: bool = False) -> dict:
    out = {f: getattr(rec, f) for f in RECORD_FIELDS}
    out.update(
        id=rec.id, document_id=doc.id, filename=doc.filename, mode=doc.mode,
        model=doc.model, status=rec.status, field_meta=rec.field_meta,
        updated_at=rec.updated_at.isoformat(), cached=cached,
    )
    return out


def llm_call(system: str, user: str, images: list, rcfg: dict | None = None) -> dict:
    """Один вызов модели + одна повторная попытка при невалидном JSON. Считает лимит."""
    fn = extract.provider_fn(rcfg or cfg)
    last = None
    for _ in range(2):
        with _lock:
            if _calls["n"] >= cfg["limits"]["max_llm_calls"]:
                raise HTTPException(429, "Достигнут лимит LLM-вызовов для этого процесса")
            _calls["n"] += 1
        try:
            return extract.parse_json(fn(rcfg or cfg, system, user, images))
        except (ValueError, json.JSONDecodeError) as e:
            last = e
    raise extract.ProviderError(f"Модель вернула невалидный JSON: {last}")


@app.get("/health")
def health():
    key = "GEMINI_API_KEY" if cfg["provider"] == "gemini" else "GROQ_API_KEY"
    return {
        "ok": True, "provider": cfg["provider"], "model": extract.model_name(cfg),
        "key_present": bool(os.getenv(key)), "llm_calls_used": _calls["n"],
        "llm_calls_max": cfg["limits"]["max_llm_calls"],
    }


@app.post("/api/documents")
def upload(
    file: UploadFile = File(...), db=Depends(get_db),
    x_provider: str | None = Header(None), x_model: str | None = Header(None),
    x_gemini_key: str | None = Header(None), x_groq_key: str | None = Header(None),
):
    # Настройки из UI: провайдер/модель/ключи на один запрос, config.yaml и .env остаются дефолтом.
    rcfg = cfg
    if x_provider in extract.PROVIDERS:
        rcfg = {**cfg, "provider": x_provider}
    if x_model and x_model.strip():
        p = rcfg["provider"]
        rcfg = {**rcfg, p: {**cfg[p], "model": x_model.strip()}}
    extract.KEYS.set({"GEMINI_API_KEY": (x_gemini_key or "").strip(), "GROQ_API_KEY": (x_groq_key or "").strip()})
    lim = cfg["limits"]
    max_bytes = lim["max_upload_mb"] * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"Файл больше {lim['max_upload_mb']} МБ")
    mime = extract.sniff_mime(data)
    if not mime:
        raise HTTPException(415, "Поддерживаются только PDF, JPG и PNG")

    sha = hashlib.sha256(data).hexdigest()
    doc = db.scalar(select(Document).where(Document.sha256 == sha))
    if doc:  # кэш: тот же файл уже обработан, LLM не вызываем
        rec = db.scalar(select(Record).where(Record.document_id == doc.id))
        return serialize(rec, doc, cached=True)

    try:
        mode, text, images = extract.prepare(data, mime, cfg)
    except ValueError as e:
        raise HTTPException(422, str(e))

    known = db.scalars(
        select(Record.organization)
        .where(Record.status == "confirmed", Record.organization.is_not(None))
        .distinct().limit(lim["known_orgs_in_prompt"])
    ).all()
    system, user = extract.build_prompts(rcfg, mode, text, list(known))

    try:
        raw = llm_call(system, user, images, rcfg)
    except extract.ProviderError as e:
        raise HTTPException(502, str(e))

    fields, meta, status = extract.clean(raw, cfg, text if mode == "text" else None)

    safe = re.sub(r"[^\w.\- ]", "_", Path(file.filename or "file").name)[:200]
    path = UPLOAD_DIR / f"{sha}.{extract.EXT[mime]}"
    path.write_bytes(data)
    doc = Document(
        sha256=sha, filename=safe, mime=mime, file_path=str(path), mode=mode,
        provider=rcfg["provider"], model=extract.model_name(rcfg),
    )
    db.add(doc)
    db.flush()
    rec = Record(document_id=doc.id, field_meta=meta, status=status, **fields)
    db.add(rec)
    db.commit()
    return serialize(rec, doc)


@app.get("/api/records")
def list_records(status: str | None = None, db=Depends(get_db)):
    q = select(Record, Document).join(Document, Record.document_id == Document.id)
    if status:
        q = q.where(Record.status == status)
    q = q.order_by(Record.id.desc()).limit(500)
    return [serialize(r, d) for r, d in db.execute(q).all()]


class RecordPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    department: str | None = None
    full_name: str | None = None
    position: str | None = None
    course_title: str | None = None
    city: str | None = None
    country_code: str | None = None
    scope: Literal["kz", "abroad", "unknown"] | None = None
    date_start: str | None = None
    date_end: str | None = None
    organization: str | None = None
    hours: int | None = None
    completion_form: str | None = None
    direction: str | None = None
    status: Literal["auto", "needs_review", "confirmed"] | None = None

    @field_validator("date_start", "date_end")
    @classmethod
    def _date(cls, v):
        if v is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("Ожидается YYYY-MM-DD")
        return v


@app.patch("/api/records/{record_id}")
def patch_record(record_id: int, body: RecordPatch, db=Depends(get_db)):
    rec = db.get(Record, record_id)
    if not rec:
        raise HTTPException(404, "Запись не найдена")
    changes = body.model_dump(exclude_unset=True)
    status = changes.pop("status", None) or "confirmed"
    meta = dict(rec.field_meta or {})
    for f, v in changes.items():
        setattr(rec, f, v)
        meta[f] = {"source": "manual"}
    rec.field_meta, rec.status = meta, status  # новый dict, чтобы SQLAlchemy заметил изменение JSON
    db.commit()
    return serialize(rec, db.get(Document, rec.document_id))


@app.get("/api/documents/{doc_id}/file")
def get_file(doc_id: int, db=Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc or not Path(doc.file_path).exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(doc.file_path, media_type=doc.mime)


# UI: http://localhost:8000/ (должно быть последним, после всех /api и /health)
UI_DIR = Path(__file__).resolve().parent / "ui"
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
