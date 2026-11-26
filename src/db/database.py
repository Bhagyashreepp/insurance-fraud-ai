"""SQLite engine and session helpers."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, event
from sqlalchemy.engine import create_engine as sqlalchemy_create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

DEFAULT_DATABASE_PATH = Path("data/insurance.db")


def create_engine(database_path: Path = DEFAULT_DATABASE_PATH) -> Engine:
    """Create a SQLite engine with foreign-key enforcement enabled."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqlalchemy_create_engine(f"sqlite:///{database_path}")

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


def create_schema(engine: Engine) -> None:
    """Create all database tables if they do not already exist."""
    Base.metadata.create_all(engine)


def reset_schema(engine: Engine) -> None:
    """Replace the database schema; used only by the reproducible seed command."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def get_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a configured SQLAlchemy session factory."""
    return sessionmaker(bind=engine, expire_on_commit=False)
