# Serving layer image.
#
# Python 3.12 rather than the 3.9 this was developed against: the code is
# 3.9-compatible (no match statements, no PEP 604 unions) but there is no
# reason to ship an interpreter three years older than necessary, and Phase 6
# pulls in torch, which is happier on modern Python.
FROM python:3.12-slim-bookworm

# PYTHONDONTWRITEBYTECODE: no .pyc clutter in the image or the volume.
# PYTHONUNBUFFERED: without it, stdout is block-buffered when not a TTY and
# `docker compose logs` shows nothing until the buffer fills — which looks
# exactly like a hung process.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies before source, so editing a .py file does not invalidate the
# pip install layer. This is the difference between a 2-second rebuild and a
# 40-second one.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app/ ./app/

# Run as a non-root user. Nothing here needs root, and Phase 4.3 puts this
# image on the public internet.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# No curl or wget in the slim image, and adding one just for a health check is
# a wasted layer plus extra attack surface — the interpreter is already here.
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
