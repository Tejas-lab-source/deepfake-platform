FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# WITH_ML=true adds the real-mode stack (PyTorch CPU, MediaPipe, ...): about 2.5 GB.
# The default image is the small mock-mode one. MediaPipe and OpenCV need the system
# libraries below; the names are for Debian 13 (trixie), which python:3.11-slim is built on.
ARG WITH_ML=false
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_ML" = "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0t64 \
      && rm -rf /var/lib/apt/lists/* \
      && pip install --no-cache-dir -r requirements-ml.txt; \
    fi

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
