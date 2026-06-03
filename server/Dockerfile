# =============================================================================
# server/Dockerfile — lightweight Flask API image.
#
# The API only enqueues Celery jobs and serves their status, so it needs just
# Flask + the Celery client (+ redis backend) — none of the heavy ML stack used
# by the segmentation worker. Project code is volume-mounted at /app at runtime.
# =============================================================================

FROM python:3.12-slim

WORKDIR /app

# Install only the API's Python dependencies.
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

EXPOSE 5005

# Flask application factory lives at server/__init__.py::create_app().
CMD ["python", "-m", "flask", "--app", "server:create_app()", "run", \
     "--host", "0.0.0.0", "--port", "5005"]
