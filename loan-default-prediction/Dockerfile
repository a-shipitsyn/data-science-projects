# API image. Train first (cd src && python train.py) so that models/model.pkl exists.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch wheel keeps the image small
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY src/ src/
COPY models/model.pkl models/model.pkl

WORKDIR /app/src
ENV MODEL_PATH=/app/models/model.pkl OMP_NUM_THREADS=1
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
