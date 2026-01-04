FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and install dependencies
COPY Pipfile Pipfile.lock* /app/
RUN pipenv install --system --deploy --ignore-pipfile

# Copy application files
COPY predict.py /app/
COPY model.pkl /app/
COPY dv.pkl /app/

# Expose port
EXPOSE 9696

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
