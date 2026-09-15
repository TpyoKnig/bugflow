FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY fly_brain/ ./fly_brain/

# Non-root, read-only-rootfs friendly: the bridge writes nothing to disk.
RUN useradd --uid 10001 --no-create-home fly
USER 10001

# Configure via env (see README "Configuration"): N8N_WEBHOOK_URL, FLY_MODE,
# FLY_STEPS (0 = forever), FLY_STEP_DELAY, NEUPRINT_TOKEN, FLY_SENSORY_GROUPS, ...
ENTRYPOINT ["python", "-m", "fly_brain.bridge"]
