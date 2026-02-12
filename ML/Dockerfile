FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

