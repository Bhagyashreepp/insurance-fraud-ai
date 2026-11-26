"""Load generated synthetic CSV files into the local SQLite database."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine

from src.data.models import ENTITY_TABLES

from .database import DEFAULT_DATABASE_PATH, create_engine, get_session_factory, reset_schema
from .models import Agent, Claim, Contractor, Customer, Document, Policy

LOGGER = logging.getLogger(__name__)
TABLE_MODELS = {
    "agents": Agent,
    "customers": Customer,
    "policies": Policy,
    "contractors": Contractor,
    "claims": Claim,
    "documents": Document,
}
DATE_COLUMNS = {
    "agents": ("hire_date",),
    "customers": ("customer_since",),
    "policies": ("effective_date", "expiration_date"),
    "claims": ("incident_date", "submitted_date"),
    "documents": ("document_date", "uploaded_date"),
}
BOOLEAN_COLUMNS = {"agents": ("active",), "contractors": ("licensed", "active")}


def _records_from_csv(path: Path, table_name: str) -> list[dict[str, Any]]:
    """Read one CSV and normalize values for SQLAlchemy bulk inserts."""
    frame = pd.read_csv(path, keep_default_na=False)
    for column in DATE_COLUMNS.get(table_name, ()):
        frame[column] = pd.to_datetime(frame[column]).dt.date
    for column in BOOLEAN_COLUMNS.get(table_name, ()):
        frame[column] = frame[column].astype(bool)
    if table_name == "claims":
        frame["duplicate_of_claim_id"] = frame["duplicate_of_claim_id"].replace("", None)
    return frame.to_dict(orient="records")


def seed_database(raw_dir: Path = Path("data/raw"), database_path: Path = DEFAULT_DATABASE_PATH) -> dict[str, int]:
    """Reset and load the six generated CSVs, returning inserted row counts."""
    missing = [name for name in ENTITY_TABLES if not (raw_dir / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing generated CSV files in {raw_dir}: {', '.join(missing)}")

    engine = create_engine(database_path)
    reset_schema(engine)
    session_factory = get_session_factory(engine)
    counts: dict[str, int] = {}
    with session_factory.begin() as session:
        for table_name in ENTITY_TABLES:
            records = _records_from_csv(raw_dir / f"{table_name}.csv", table_name)
            session.execute(insert(TABLE_MODELS[table_name]), records)
            counts[table_name] = len(records)
    return counts


def database_row_counts(engine: Engine) -> dict[str, int]:
    """Return persisted row counts, useful for seed verification."""
    session_factory = get_session_factory(engine)
    with session_factory() as session:
        return {table_name: int(session.scalar(select(func.count()).select_from(model)) or 0) for table_name, model in TABLE_MODELS.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load synthetic insurance CSVs into SQLite.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory containing generated CSV files.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH, help="SQLite database output path.")
    return parser.parse_args()


def main() -> None:
    """Run the SQLite seed command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    counts = seed_database(args.raw_dir, args.database)
    LOGGER.info("Seeded %s", args.database)
    LOGGER.info("Row counts: %s", counts)


if __name__ == "__main__":
    main()
