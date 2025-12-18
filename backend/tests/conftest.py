from collections.abc import Iterator
from http import HTTPStatus

import pytest
from database import engine
from fastapi.testclient import TestClient
from main import app
from models import Base
from sqlalchemy import text


@pytest.fixture(scope="session", autouse=True)
def _create_tables_once() -> Iterator[None]:
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(autouse=True)
def _clean_db() -> Iterator[None]:
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM items"))
        conn.execute(text("DELETE FROM users"))
    yield


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_token(client: TestClient) -> str:
    email = "tester@example.com"
    pw = "secretpw"
    r = client.post("/auth/signup", json={"email": email, "password": pw})
    assert r.status_code in (HTTPStatus.CREATED, HTTPStatus.BAD_REQUEST)
    r = client.post("/auth/signin", json={"email": email, "password": pw})
    assert r.status_code == HTTPStatus.OK, r.text
    return r.json()["access_token"]


@pytest.fixture()
def auth_headers(auth_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth_token}"}
