from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError


def _import(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


# Prefer underscored modules if present, otherwise fall back.
schemas_ = _import("schemas_") or importlib.import_module("schemas")
security_ = _import("security_") or importlib.import_module("security")


def test_schemas_usercreate_valid_email() -> None:
    u = schemas_.UserCreate(email="a@example.com", password="pw1234")  # len==6
    assert u.email == "a@example.com"
    assert u.password == "pw1234"


def test_schemas_usercreate_invalid_email_raises() -> None:
    with pytest.raises(ValidationError) as ex:
        schemas_.UserCreate(email="not-an-email", password="pw1234")  # keep pw valid
    # ensure the error is about email (not password)
    assert any(err.get("loc", [])[-1] == "email" for err in ex.value.errors())


def test_security_hash_and_verify_password() -> None:
    hashed = security_.hash_password("pw1234")
    assert security_.verify_password("pw1234", hashed) is True
    assert security_.verify_password("wrongpw", hashed) is False


def test_security_token_roundtrip_and_invalid_token() -> None:
    token = security_.create_access_token("user@example.com")
    assert security_.decode_token(token) == "user@example.com"
    assert security_.decode_token("this.is.not.jwt") is None
