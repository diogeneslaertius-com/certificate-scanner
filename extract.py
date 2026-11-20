"""Файл -> (PDF-текст | картинки) -> LLM -> проверенный dict.

Провайдер это обычная функция (cfg, system, user, images) -> str, поэтому
в тестах её подменяют заглушкой, а в проде выбирают через config.yaml.
"""
import base64
import json
import os
import re
from contextvars import ContextVar
from datetime import date

import httpx
import pymupdf
import yaml
from dotenv import load_dotenv

from db import BASE_DIR

load_dotenv(BASE_DIR / ".env")

MAGIC = {b"%PDF": "application/pdf", b"\x89PNG": "image/png", b"\xff\xd8\xff": "image/jpeg"}
EXT = {"application/pdf": "pdf", "image/png": "png", "image/jpeg": "jpg"}
REQUIRED = ["full_name", "course_title", "organization", "hours", "date_start"]
LLM_FIELDS = [
    "full_name", "course_title", "organization", "city", "country_code",
    "date_start", "date_end", "hours", "completion_form", "direction",
]


# Ключи из заголовков запроса (UI). Приоритет над .env, живут только в рамках одного запроса.
KEYS: ContextVar[dict] = ContextVar("keys", default={})


class ProviderError(Exception):
    pass


def load_config() -> dict:
    cfg = yaml.safe_load((BASE_DIR / "config.yaml").read_text(encoding="utf-8"))
    cfg["provider"] = os.getenv("CERTSCAN_PROVIDER", cfg["provider"])
    return cfg


# ------------------------------------------------------------------ ingest

def sniff_mime(data: bytes) -> str | None:
    for sig, mime in MAGIC.items():
        if data.startswith(sig):
            return mime
    return None


def good_text_layer(text: str, pages: int, lim: dict) -> bool:
    """Текстовый слой годится, если текста достаточно и это в основном буквы
    (битая кодировка шрифта даёт мусор из спецсимволов)."""
    chars = re.sub(r"\s", "", text)
    if len(chars) < lim["min_text_chars_per_page"] * pages:
        return False
    letters = sum(c.isalpha() for c in chars)
    return letters / len(chars) >= lim["min_letter_ratio"]


def to_jpeg(pix: "pymupdf.Pixmap", lim: dict) -> tuple[bytes, str]:
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)                    # убрать альфа-канал
    if pix.colorspace is None or pix.colorspace.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)        # gray/cmyk -> rgb
    while max(pix.width, pix.height) > lim["max_image_side"]:
        pix.shrink(1)                                   # уменьшение вдвое
    return pix.tobytes("jpeg", jpg_quality=85), "image/jpeg"


def prepare(data: bytes, mime: str, cfg: dict) -> tuple[str, str, list]:
    """-> (mode, text, images[(bytes, mime)]). ValueError для плохих файлов."""
    lim = cfg["limits"]
    try:
        if mime == "application/pdf":
            doc = pymupdf.open(stream=data, filetype="pdf")
            if doc.page_count == 0:
                raise ValueError("Пустой PDF")
            if doc.page_count > lim["max_pdf_pages"]:
                raise ValueError(f"В PDF больше {lim['max_pdf_pages']} страниц")
            text = "\n".join(p.get_text() for p in doc).strip()
            if good_text_layer(text, doc.page_count, lim):
                return "text", text[: lim["max_text_chars"]], []
            imgs = [to_jpeg(p.get_pixmap(dpi=lim["render_dpi"]), lim) for p in doc]
            return "image", "", imgs
        return "image", "", [to_jpeg(pymupdf.Pixmap(data), lim)]
    except ValueError:
        raise
    except Exception as e:  # битый файл
        raise ValueError(f"Не удалось прочитать файл: {e}") from e


# ----------------------------------------------------------------- prompts

def build_prompts(cfg: dict, mode: str, text: str, known_orgs: list[str]) -> tuple[str, str]:
    p = cfg["prompts"]
    system = p["system"].replace("{{directions}}", "; ".join(f'"{d}"' for d in cfg["directions"]))
    if cfg["provider"] == "gemini" and cfg["gemini"].get("use_search"):
        system += "\n" + p["search_addendum"]
    user = p["user_text"].replace("{{text}}", text) if mode == "text" else p["user_image"]
    parts = [user, p["schema"]]
    if known_orgs:
        parts.insert(1, p["known_orgs"].replace("{{known_orgs}}", "\n".join(f"- {o}" for o in known_orgs)))
    return system, "\n\n".join(parts)


# --------------------------------------------------------------- providers

def _key(name: str) -> str:
    k = KEYS.get().get(name) or os.getenv(name)
    if not k:
        raise ProviderError(f"Не задан {name} (см. .env.example)")
    return k


def _post(url: str, headers: dict, body: dict, timeout: int) -> dict:
    try:
        r = httpx.post(url, headers=headers, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise ProviderError(f"HTTP {e.response.status_code}: {e.response.text[:300]}") from None
    except httpx.HTTPError as e:
        raise ProviderError(f"Сеть: {type(e).__name__}") from None


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def call_groq(cfg, system, user, images) -> str:
    g = cfg["groq"]
    if images:
        content = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": f"data:{m};base64,{_b64(b)}"}}
            for b, m in images
        ]
    else:
        content = user
    body = {
        "model": g["model"],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        **g.get("extra", {}),
    }
    data = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {_key('GROQ_API_KEY')}"},
        body, cfg["limits"]["timeout_s"],
    )
    return data["choices"][0]["message"]["content"] or ""


def call_gemini(cfg, system, user, images) -> str:
    g = cfg["gemini"]
    parts = [{"text": user}] + [
        {"inline_data": {"mime_type": m, "data": _b64(b)}} for b, m in images
    ]
    # temperature намеренно не задаём: для Gemini 3.x рекомендуют значение по умолчанию.
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", **g.get("generation_extra", {})},
    }
    if g.get("use_search"):
        body["tools"] = [{"google_search": {}}]
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{g['model']}:generateContent",
        {"x-goog-api-key": _key("GEMINI_API_KEY")},
        body, cfg["limits"]["timeout_s"],
    )
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        raise ProviderError(f"Пустой ответ модели: {str(data)[:300]}") from None
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


PROVIDERS = {"groq": call_groq, "gemini": call_gemini}


def provider_fn(cfg: dict):
    try:
        return PROVIDERS[cfg["provider"]]
    except KeyError:
        raise ProviderError(f"Неизвестный провайдер: {cfg['provider']}") from None


def model_name(cfg: dict) -> str:
    return cfg[cfg["provider"]]["model"]


# ----------------------------------------------------------- parse + clean

def parse_json(s: str) -> dict:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("В ответе нет JSON")
    obj = json.loads(s[i : j + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON не объект")
    return obj


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).casefold().replace("ё", "е").strip()


def _str(v):
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return None


def _iso(v):
    try:
        return date.fromisoformat(v).isoformat() if isinstance(v, str) else None
    except ValueError:
        return None


def _hours(v):
    if isinstance(v, str) and re.search(r"ects|кредит|credit", v, re.I):
        return None                       # кредиты это не часы
    try:
        n = int(float(str(v).strip().split()[0]))
    except (ValueError, IndexError):
        return None
    return n if 1 <= n <= 2000 else None


def clean(raw: dict, cfg: dict, text: str | None) -> tuple[dict, dict, str]:
    """Валидирует ответ модели. -> (поля, field_meta, status).

    text передаём только в text-режиме: тогда проверяем, что цитата-доказательство
    действительно есть в документе (verified). Для картинок проверить нечем -> None.
    """
    ev = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
    rec = {f: _str(raw.get(f)) for f in LLM_FIELDS}
    rec["hours"] = _hours(raw.get("hours"))
    rec["date_start"], rec["date_end"] = _iso(raw.get("date_start")), _iso(raw.get("date_end"))
    if rec["date_start"] and not rec["date_end"]:
        rec["date_end"] = rec["date_start"]

    cc = (rec["country_code"] or "").upper()
    rec["country_code"] = cc if re.fullmatch(r"[A-Z]{2}", cc) else None
    rec["scope"] = "unknown" if not rec["country_code"] else ("kz" if rec["country_code"] == "KZ" else "abroad")
    if rec["direction"] not in cfg["directions"]:
        rec["direction"] = "Другое"
    if rec["completion_form"]:
        rec["completion_form"] = rec["completion_form"].casefold()

    haystack = _squash(text) if text is not None else None
    meta, issues = {}, []
    for f in LLM_FIELDS:
        if rec[f] is None:
            continue
        quote = _str(ev.get(f))
        verified = None
        if haystack is not None:
            verified = bool(quote) and _squash(quote) in haystack
        meta[f] = {"source": "llm", "evidence": quote, "verified": verified}
        if verified is False and f in REQUIRED:
            issues.append(f)
    if rec["date_start"] and rec["date_end"] and rec["date_end"] < rec["date_start"]:
        issues.append("dates")
    meta["_issues"] = issues

    missing = [f for f in REQUIRED if rec.get(f) is None]
    status = "needs_review" if (missing or issues) else "auto"
    return rec, meta, status
