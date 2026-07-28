"""
Async SQLAlchemy engine/session setup for the SafeSignal incidents store.

Replaces the write-only local_storage.py / S3-Excel archives with a real,
queryable DB. SQLite by default (instant local validation, zero setup) --
swap DATABASE_URL to a Postgres asyncpg URL later with no code change.
"""
import os
from contextlib import asynccontextmanager

from sqlalchemy import Integer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite+aiosqlite:///{DATA_DIR}/safesignal.db")

engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def _add_missing_columns(conn) -> None:
    """
    Lightweight ad-hoc migration: create_all only creates tables that don't
    exist yet, it never alters an existing one, so a column added to models.py
    after the incidents table already has rows on disk (e.g. tokens_used)
    would otherwise silently never appear. No Alembic in this project, so
    this just diffs PRAGMA table_info against the ORM's declared columns and
    ALTERs in whatever's missing -- safe to run on every startup.
    """
    from sqlalchemy import text

    from models import Base

    for table in Base.metadata.tables.values():
        existing = {row[1] for row in (await conn.execute(text(f'PRAGMA table_info("{table.name}")'))).fetchall()}
        for column in table.columns:
            if column.name in existing:
                continue
            col_type = column.type.compile(engine.dialect)
            if column.nullable:
                default = "NULL"
            else:
                default = "0" if isinstance(column.type, Integer) else "''"
            await conn.execute(text(
                f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type} DEFAULT {default}'
            ))


async def init_db() -> None:
    """Creates data/ and all tables on startup if they don't already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    from models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _add_missing_columns(conn)


@asynccontextmanager
async def get_session():
    """Standalone session context manager -- used inside route handlers that
    aren't wired through FastAPI's Depends() (the DB write happens deep inside
    existing merged()-style handlers, not as a top-level dependency)."""
    async with AsyncSessionLocal() as session:
        yield session


def upsert_stmt(model, *, index_elements: list[str], values: dict):
    """
    Dialect-aware insert-or-update. SQLite and Postgres each need their own
    ON CONFLICT construct (there's no dialect-agnostic upsert in SQLAlchemy
    core) -- picking one based on the live engine's dialect is what actually
    makes DATABASE_URL swappable to Postgres later, rather than only
    appearing to be.
    """
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as dialect_insert
    else:
        from sqlalchemy.dialects.sqlite import insert as dialect_insert

    stmt = dialect_insert(model).values(**values)
    return stmt.on_conflict_do_update(index_elements=index_elements, set_=values)
