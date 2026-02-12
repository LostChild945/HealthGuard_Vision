FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier requirements en premier pour le cache
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Utiliser shell form pour interpolation
CMD /opt/venv/bin/uvicorn API.src.app:main --host 0.0.0.0 --port ${API_PORT}
