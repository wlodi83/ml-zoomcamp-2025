# Workflow Execution Duration Prediction - Docker Image
# Build: docker build -t duration-prediction .
# Run:   docker run -p 9696:9696 duration-prediction

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY predict.py .
COPY model.xgb .
COPY dv.pkl .
COPY model_info.pkl .

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=9696

# Expose port
EXPOSE 9696

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/health')" || exit 1

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "--workers", "2", "--timeout", "120", "predict:app"]
