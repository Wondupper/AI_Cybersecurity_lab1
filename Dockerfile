# Образ выполняет одну проверку (smoke): сверяет хэши модели, загружает её и преобразует тестовый WAV.
# Базовый образ закреплён версией. Точный digest добавляется после `docker pull` (см. README).

# --- этап сборки: компилятор нужен только для pyworld (колеса под Python 3.14 нет)
FROM python:3.14.4-slim-trixie AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements-runtime.txt .
RUN pip wheel --wheel-dir /wheels -r requirements-runtime.txt

# --- рабочий образ: только runtime-зависимости, без компилятора, pytest и Jupyter
FROM python:3.14.4-slim-trixie
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-index --find-links=/wheels -r /tmp/requirements-runtime.txt \
    && rm /tmp/requirements-runtime.txt
WORKDIR /app
# в образ включаются только нужные файлы; модель добавляется явно
COPY src/predict.py src/gmm_vc.py src/checksums.py src/smoke.py ./src/
COPY models/gmm_M16.npz models/f0_stats.json models/SHA256SUMS ./models/
USER 10001
# контейнер выполняет одну команду и завершается, поэтому проверка состояния не нужна
HEALTHCHECK NONE
CMD ["python", "src/smoke.py"]
