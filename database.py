"""
Async SQLAlchemy engine/session setup for the SafeSignal incidents store.

Replaces the write-only local_storage.py / S3-Excel archives with a real,
queryable DB. SQLite by default (instant local validation, zero setup) --
swap DATABASE_URL to a Postgres asyncpg URL later with no code change.
"""
import os
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite+aiosqlite:///{DATA_DIR}/safesignal.db")

engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """Creates data/ and all tables on startup if they don't already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    from models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


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
