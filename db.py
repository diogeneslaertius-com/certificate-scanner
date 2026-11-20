import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# Одна точка, откуда берутся пути (потом сюда добавится sys._MEIPASS для exe).
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("CERTSCAN_DATA", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DATA_DIR / 'certscan.db'}",
    connect_args={"check_same_thread": False},
)
Session = sessionmaker(engine, expire_on_commit=False)

# Поля записи в порядке таблицы из ТЗ. department/position модель не заполняет.
RECORD_FIELDS = [
    "department", "full_name", "position", "course_title", "city", "country_code",
    "scope", "date_start", "date_end", "organization", "hours",
    "completion_form", "direction",
]


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    mime: Mapped[str] = mapped_column(String(50))
    file_path: Mapped[str] = mapped_column(String(500))
    mode: Mapped[str] = mapped_column(String(10))        # text | image
    provider: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Record(Base):
    __tablename__ = "records"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    department: Mapped[str | None] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(255))
    position: Mapped[str | None] = mapped_column(String(255))
    course_title: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(String(120))
    country_code: Mapped[str | None] = mapped_column(String(2))
    scope: Mapped[str] = mapped_column(String(10), default="unknown")  # kz | abroad | unknown
    date_start: Mapped[str | None] = mapped_column(String(10))
    date_end: Mapped[str | None] = mapped_column(String(10))
    organization: Mapped[str | None] = mapped_column(String(255))
    hours: Mapped[int | None] = mapped_column(Integer)
    completion_form: Mapped[str | None] = mapped_column(String(30))
    direction: Mapped[str | None] = mapped_column(String(255))
    field_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(15), default="needs_review")  # auto | needs_review | confirmed
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


def init_db():
    Base.metadata.create_all(engine)
