import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlmodel import select

from app.auth import (
    AdminUser,
    CurrentUser,
    SessionDep,
    create_token,
    hash_password,
    verify_password,
)
from app.models import User, UserPreference, now

router = APIRouter(prefix="/auth", tags=["auth"])


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8)
    is_admin: bool = False


class UserRead(BaseModel):
    id: int
    username: str
    is_admin: bool


@router.post("/login")
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()], session: SessionDep
) -> Token:
    user = session.exec(select(User).where(User.username == form.username)).first()
    if user is None or not verify_password(form.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username or password")
    return Token(access_token=create_token(user))


@router.get("/me")
def me(user: CurrentUser) -> UserRead:
    return UserRead.model_validate(user, from_attributes=True)


@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(body: UserCreate, session: SessionDep, _: AdminUser) -> UserRead:
    if session.exec(select(User).where(User.username == body.username)).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "Username already taken")
    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        is_admin=body.is_admin,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return UserRead.model_validate(user, from_attributes=True)


@router.get("/users")
def list_users(session: SessionDep, _: CurrentUser) -> list[UserRead]:
    users = session.exec(select(User).order_by(User.username)).all()
    return [UserRead.model_validate(u, from_attributes=True) for u in users]


PREFERENCES_MAX_BYTES = 32 * 1024


@router.get("/me/preferences")
def get_preferences(user: CurrentUser, session: SessionDep) -> dict[str, Any]:
    pref = session.get(UserPreference, user.id)
    return pref.data if pref else {}


@router.patch("/me/preferences")
def update_preferences(
    patch: dict[str, Any], user: CurrentUser, session: SessionDep
) -> dict[str, Any]:
    """Merge top-level keys into the user's preferences; a null value removes a key."""
    assert user.id is not None
    pref = session.get(UserPreference, user.id) or UserPreference(user_id=user.id)
    data = {**pref.data, **patch}
    data = {k: v for k, v in data.items() if v is not None}
    if len(json.dumps(data)) > PREFERENCES_MAX_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, "Preferences are too large"
        )
    pref.data = data  # reassign so SQLAlchemy sees the JSON change
    pref.updated_at = now()
    session.add(pref)
    session.commit()
    return data
