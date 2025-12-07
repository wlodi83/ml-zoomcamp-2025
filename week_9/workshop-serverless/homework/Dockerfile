FROM agrigorev/model-2025-hairstyle:v1

# Install required Python packages
RUN pip install --no-cache-dir \
    onnxruntime \
    pillow \
    numpy

# Copy lambda function code
COPY lambda_function.py .
RUN chmod 644 lambda_function.py

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]