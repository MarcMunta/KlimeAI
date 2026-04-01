FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NPM_CONFIG_UPDATE_NOTIFIER=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
    && npm install -g pnpm \
    && python -m pip install --no-cache-dir \
        uv \
        pytest \
        httpx \
        fastapi \
        semgrep \
        bandit \
        pip-audit \
        ruff \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
