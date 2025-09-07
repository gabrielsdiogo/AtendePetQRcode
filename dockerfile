FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependência nativa para pyzbar (barcodes 1D)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# usuário não-root
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
ENV PORT=8000 UVICORN_WORKERS=2

CMD sh -c 'uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}'
