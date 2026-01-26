FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /repo

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
COPY requirements.txt ./
COPY requirements.test.txt ./
RUN python -m pip install --upgrade pip \
 && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && python -m pip install --no-cache-dir -r requirements.txt -r requirements.test.txt

