FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar app
COPY app.py .

# Usuário não-root (boa prática)
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Permite configurar porta e workers via env
ENV PORT=8000 \
    UVICORN_WORKERS=2

CMD sh -c 'uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}'
