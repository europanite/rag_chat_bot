from http import HTTPStatus

from fastapi.testclient import TestClient


def test_signup_success(client: TestClient):
    r = client.post(
        "/auth/signup", json={"email": "u1@example.com", "password": "123456"}
    )
    assert r.status_code == HTTPStatus.CREATED, r.text
    data = r.json()
    assert data["email"] == "u1@example.com"
    assert "id" in data


def test_signup_short_password(client):
    r = client.post("/auth/signup", json={"email": "u2@example.com", "password": "123"})
    assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    detail = r.json().get("detail", [])
    assert any(
        (d.get("loc")[-1] == "password") and ("string_too_short" in d.get("type", ""))
        for d in detail
    )


def test_signup_duplicate_email(client: TestClient):
    client.post("/auth/signup", json={"email": "u3@example.com", "password": "123456"})
    r = client.post(
        "/auth/signup", json={"email": "u3@example.com", "password": "abcdef"}
    )
    assert r.status_code == HTTPStatus.BAD_REQUEST
    assert "Email already registered" in r.text


def test_signin_ok(client: TestClient):
    client.post("/auth/signup", json={"email": "u4@example.com", "password": "abcdef"})
    r = client.post(
        "/auth/signin", json={"email": "u4@example.com", "password": "abcdef"}
    )
    assert r.status_code == HTTPStatus.OK
    token = r.json().get("access_token")
    assert token and isinstance(token, str)


def test_signin_invalid(client: TestClient):
    r = client.post(
        "/auth/signin", json={"email": "nope@example.com", "password": "xxxxx"}
    )
    assert r.status_code == HTTPStatus.UNAUTHORIZED


def test_me_requires_token(client: TestClient):
    r = client.get("/auth/me")
    assert r.status_code == HTTPStatus.UNAUTHORIZED


def test_me_ok(client: TestClient, auth_headers):
    r = client.get("/auth/me", headers=auth_headers)
    assert r.status_code == HTTPStatus.OK
    assert "email" in r.json()
