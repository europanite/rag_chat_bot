from http import HTTPStatus

import security
from fastapi.testclient import TestClient


# 1) double sign up -> 409 or 400
def test_signup_duplicate_email(client: TestClient):
    email = "dup@example.com"
    pw = "abcdef"  # min length
    r1 = client.post("/auth/signup", json={"email": email, "password": pw})
    assert r1.status_code in (HTTPStatus.OK, HTTPStatus.CREATED)

    r2 = client.post("/auth/signup", json={"email": email, "password": pw})
    assert r2.status_code in (HTTPStatus.BAD_REQUEST, HTTPStatus.CONFLICT)


# 2) password inconsistent -> 401
def test_signin_wrong_password(client: TestClient):
    email = "wrongpw@example.com"
    ok_pw = "abcdef"
    client.post("/auth/signup", json={"email": email, "password": ok_pw})

    r = client.post("/auth/signin", json={"email": email, "password": "not-correct"})
    assert r.status_code == HTTPStatus.UNAUTHORIZED


# 3) bad token /auth/me -> 401: JWT decode failure
def test_me_invalid_token(client: TestClient):
    r = client.get("/auth/me", headers={"Authorization": "Bearer not-a-token"})
    assert r.status_code == HTTPStatus.UNAUTHORIZED


# 4) expired token /auth/me -> 401: exp check
def test_me_expired_token(client: TestClient, monkeypatch):
    email = "expired@example.com"
    pw = "abcdef"
    client.post("/auth/signup", json={"email": email, "password": pw})

    original = security.EXPIRE_MIN
    monkeypatch.setattr(security, "EXPIRE_MIN", -1)
    try:
        token = security.create_access_token(email)
    finally:
        monkeypatch.setattr(security, "EXPIRE_MIN", original)

    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == HTTPStatus.UNAUTHORIZED
