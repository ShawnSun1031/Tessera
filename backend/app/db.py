"""Database engine and session dependency."""

from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from sqlalchemy import Engine, event
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import Session, SQLModel, create_engine, select

from app.config import get_settings


def sqlite_pragmas(dbapi_conn: Any, _: Any) -> None:  # noqa: ANN401
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=10000")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    if url.startswith("sqlite:///"):
        get_settings().storage_dir.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False})
        # API and worker write concurrently: WAL + busy timeout avoid lock errors
        event.listen(engine, "connect", sqlite_pragmas)
        return engine
    return create_engine(url)


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session


def init_db(engine: Engine) -> None:
    from app import models
    from app.auth import hash_password

    # API and worker may start together: tolerate losing either race
    try:
        SQLModel.metadata.create_all(engine)
    except OperationalError:
        SQLModel.metadata.create_all(engine)
    settings = get_settings()
    with Session(engine) as session:
        # bootstrap the first admin from the environment
        if settings.admin_password and not session.exec(select(models.User)).first():
            session.add(
                models.User(
                    username=settings.admin_username,
                    password_hash=hash_password(settings.admin_password),
                    is_admin=True,
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
