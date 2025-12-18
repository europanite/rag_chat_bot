import datetime as dt
import os

from jose import jwt
from passlib.context import CryptContext

SECRET_KEY = os.getenv("AUTH_SECRET", "dev-secret")
ALGO = "HS256"
EXPIRE_MIN = int(os.getenv("AUTH_EXPIRE_MINUTES", "60"))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    return pwd_ctx.verify(pw, hashed)


def create_access_token(sub: str) -> str:
    now = dt.datetime.now(dt.UTC)
    payload = {"sub": sub, "iat": now, "exp": now + dt.timedelta(minutes=EXPIRE_MIN)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGO)


def decode_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGO])
        return payload.get("sub")
    except Exception:
        return None
