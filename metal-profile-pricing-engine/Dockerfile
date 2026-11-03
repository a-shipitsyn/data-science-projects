FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
ENV PYTHONPATH=/app/src PRICING_STORE=/data/store
RUN useradd --create-home app && mkdir -p /data/store && chown app /data/store
USER app
VOLUME /data/store
EXPOSE 8000
# PRICING_MASTER_KEY must be passed at run time (docker run -e / a secrets manager), never baked in
CMD ["python", "-m", "pricing_engine", "serve", "--host", "0.0.0.0"]
