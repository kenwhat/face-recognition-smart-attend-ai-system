FROM python:3.11-slim

# system libraries OpenCV (libgl1, libglib2.0-0) and pyttsx3 (espeak) need on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 espeak && rm -rf /var/lib/apt/lists/*

ENV TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p registered_faces captured_faces

# Render gives the port in $PORT (default 10000 if it is missing).
# ONE worker on purpose: the app keeps "marked today" and its locks in memory.
CMD gunicorn -b 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 300 app:app