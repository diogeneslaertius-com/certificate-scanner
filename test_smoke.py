import json
import os
import tempfile

os.environ["CERTSCAN_DATA"] = tempfile.mkdtemp()

import pymupdf
import pytest
from fastapi.testclient import TestClient

import app as appmod
import extract
from db import Base, engine

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CERT_TEXT = (
    "СЕРТИФИКАТ\nНастоящий сертификат выдан Тестов Тест Тестович\n"
    "о прохождении курса повышения квалификации по программе «Основы машинного обучения»\n"
    "в объёме 72 академических часа с 03.03.2025 по 14.03.2025\n"
    "ТОО «Пример Академия», г. Астана\n"
) * 3  # дублируем, чтобы пройти порог «достаточно текста»

GOOD = {
    "full_name": "Тестов Тест Тестович", "course_title": "Основы машинного обучения",
    "organization": "ТОО «Пример Академия»", "city": "Астана", "country_code": "kz",
    "date_start": "2025-03-03", "date_end": "2025-03-14", "hours": 72,
    "completion_form": "Сертификат", "direction": "Искусственный интеллект",
    "evidence": {"full_name": "выдан Тестов Тест Тестович", "course_title": "Основы машинного обучения",
                 "organization": "ТОО «Пример Академия»", "hours": "72 академических часа",
                 "date_start": "с 03.03.2025 по 14.03.2025"},
}


def text_pdf(text=CERT_TEXT) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text, fontsize=11,
                        fontname="dj", fontfile=FONT)
    return doc.tobytes()


def scan_png() -> bytes:
    """PDF с текстом -> картинка: имитация скана без текстового слоя."""
    pix = pymupdf.open(stream=text_pdf(), filetype="pdf")[0].get_pixmap(dpi=100)
    return pix.tobytes("png")


class Fake:
    def __init__(self, reply=GOOD):
        self.reply, self.calls = reply, []

    def __call__(self, cfg, system, user, images):
        self.calls.append((system, user, images))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply, ensure_ascii=False)


@pytest.fixture
def client(monkeypatch):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    appmod._calls["n"] = 0
    fake = Fake()
    monkeypatch.setitem(extract.PROVIDERS, appmod.cfg["provider"], fake)
    c = TestClient(appmod.app)
    c.fake = fake
    return c


def post(c, data, name="c.pdf"):
    return c.post("/api/documents", files={"file": (name, data)})


def test_text_pdf_flow_and_cache(client):
    pdf = text_pdf()
    r = post(client, pdf)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["mode"] == "text" and j["status"] == "auto"
    assert j["full_name"] == "Тестов Тест Тестович" and j["hours"] == 72
    assert j["scope"] == "kz" and j["country_code"] == "KZ"
    assert j["completion_form"] == "сертификат"
    assert j["field_meta"]["hours"]["verified"] is True
    assert client.fake.calls[0][2] == []                      # картинок нет
    assert "Тестов Тест Тестович" in client.fake.calls[0][1]  # текст ушёл в промпт

    again = post(client, pdf).json()
    assert again["cached"] is True and len(client.fake.calls) == 1


def test_image_flow(client):
    j = post(client, scan_png(), "scan.png").json()
    assert j["mode"] == "image"
    assert len(client.fake.calls[0][2]) == 1 and client.fake.calls[0][2][0][1] == "image/jpeg"
    assert j["field_meta"]["hours"]["verified"] is None       # проверить нечем
    assert j["status"] == "auto"


def test_scanned_pdf_falls_back_to_images(client):
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), stream=scan_png())
    j = post(client, doc.tobytes()).json()
    assert j["mode"] == "image"


def test_hallucinated_evidence_flags_review(client):
    bad = dict(GOOD, evidence=dict(GOOD["evidence"], hours="500 часов"))
    client.fake.reply = bad
    j = post(client, text_pdf()).json()
    assert j["field_meta"]["hours"]["verified"] is False
    assert j["status"] == "needs_review"


def test_value_cleaning(client):
    client.fake.reply = {
        "full_name": "  ", "hours": "72 ECTS", "date_start": "15.09.2026", "date_end": None,
        "country_code": "Казахстан", "direction": "Кулинария", "evidence": "oops",
    }
    j = post(client, text_pdf()).json()
    assert j["full_name"] is None and j["date_start"] is None
    assert j["hours"] is None          # "72 ECTS" это кредиты, а не часы
    assert j["scope"] == "unknown" and j["direction"] == "Другое"
    assert j["status"] == "needs_review"


def test_garbage_json_retries_once_then_502(client):
    client.fake.reply = "извините, не могу"
    r = post(client, text_pdf())
    assert r.status_code == 502 and len(client.fake.calls) == 2
    assert client.get("/api/records").json() == []            # ничего не сохранилось


def test_provider_error_not_saved(client):
    client.fake.reply = extract.ProviderError("HTTP 429")
    assert post(client, text_pdf()).status_code == 502
    assert client.get("/api/records").json() == []


def test_think_and_fences_are_stripped():
    raw = '<think>hmm {not json}</think>\n```json\n{"a": 1}\n```'
    assert extract.parse_json(raw) == {"a": 1}


def test_bad_uploads(client):
    assert post(client, b"hello", "x.txt").status_code == 415
    assert post(client, b"%PDF-1.4 broken").status_code == 422
    big = b"%PDF" + b"0" * (11 * 1024 * 1024)
    assert post(client, big).status_code == 413
    assert client.fake.calls == []


def test_call_limit(client, monkeypatch):
    monkeypatch.setitem(appmod.cfg["limits"], "max_llm_calls", 1)
    assert post(client, text_pdf()).status_code == 200
    assert post(client, text_pdf(CERT_TEXT + "другой файл")).status_code == 429


def test_patch_marks_manual_and_confirmed(client):
    rid = post(client, text_pdf()).json()["id"]
    r = client.patch(f"/api/records/{rid}", json={"department": "Кафедра ИТ", "hours": 80})
    j = r.json()
    assert j["department"] == "Кафедра ИТ" and j["hours"] == 80 and j["status"] == "confirmed"
    assert j["field_meta"]["hours"] == {"source": "manual"}
    assert client.patch(f"/api/records/{rid}", json={"date_start": "1.1.2025"}).status_code == 422
    assert client.patch(f"/api/records/{rid}", json={"bogus": 1}).status_code == 422
    assert client.patch("/api/records/999", json={"hours": 1}).status_code == 404


def test_confirmed_orgs_go_into_next_prompt(client):
    rid = post(client, text_pdf()).json()["id"]
    client.patch(f"/api/records/{rid}", json={"organization": "Назарбаев Университет"})
    post(client, text_pdf(CERT_TEXT + "второй"))
    assert "Назарбаев Университет" in client.fake.calls[1][1]


def test_file_endpoint_and_health(client):
    j = post(client, text_pdf()).json()
    assert client.get(f"/api/documents/{j['document_id']}/file").content.startswith(b"%PDF")
    h = client.get("/health").json()
    assert h["ok"] and "key" not in json.dumps(h).lower().replace("key_present", "")
