# The API image: the plane, the PyTorch worker and the small model's weights. CPU wheel here; a GPU image installs
# the CUDA wheel instead and runs on the GKE GPU pools (deploy/helm).
FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt
COPY tcs ./tcs
COPY config ./config
COPY models ./models
RUN useradd --uid 10001 --create-home tcs && mkdir -p /var/lib/tcs && chown tcs /var/lib/tcs
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "tcs.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
