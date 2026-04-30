# multi-stage so build deps don't bloat runtime
FROM python:3.12-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# torch cpu wheel only, no cuda
COPY requirements.txt .
RUN pip install --no-cache-dir --user \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt


FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY . .
RUN python manage.py collectstatic --noinput

ENV DATABASE_PATH=/data/db.sqlite3 \
    DJANGO_SETTINGS_MODULE=techpack_project.settings \
    DEBUG=0 \
    ALLOWED_HOSTS=*

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn techpack_project.wsgi:application --bind 0.0.0.0:8000 --workers 2 --timeout 120"]
