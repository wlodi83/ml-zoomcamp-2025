#
#    Dockerfile for deploying the model with Uvicorn server
#    Build command: docker build --platform=linux/amd64 -t 05-predict-conversion .
#    Run command: docker run --platform=linux/amd64 -it --rm -p 9696:9696 05-predict-conversion


FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --upgrade pip setuptools \
 && pip install uv \
 && uv sync --locked

ENV PATH="/app/.venv/bin:${PATH}"

RUN ln -s /code/pipeline_v2.bin /app/pipeline_v2.bin

COPY predict.py pipeline_v1.bin ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]