FROM python:3.11.13-slim

RUN python -m pip install --upgrade pip setuptools

RUN python -m venv /app/venv

WORKDIR /app

COPY requirements.txt .
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["/app/venv/bin/gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
