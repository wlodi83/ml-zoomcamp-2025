import onnxruntime as ort
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image


# Model path - use the one in Docker image
# For local testing, use "hair_classifier_v1.onnx"
# For Docker, the image contains "hair_classifier_empty.onnx"
try:
    # Try Docker model first
    onnx_model_path = "hair_classifier_empty.onnx"
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
except:
    # Fallback to local model for testing
    onnx_model_path = "hair_classifier_v1.onnx"
    session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Get input and output names
inputs = session.get_inputs()
outputs = session.get_outputs()
input_name = inputs[0].name
output_name = outputs[0].name


def download_image(url):
    """Download image from URL"""
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size=(200, 200)):
    """Resize image to target size"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(img):
    """
    Preprocess image for model inference.

    Steps (from homework 8):
    1. Convert to numpy array
    2. Normalize from [0, 255] to [0, 1]
    3. Apply ImageNet normalization
    4. Transpose from HWC to CHW format
    5. Add batch dimension
    6. Convert to float32
    """
    # Convert PIL Image to numpy array
    x = np.array(img, dtype='float32')

    # Normalize from [0, 255] to [0, 1]
    x = x / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (x - mean) / std

    # Transpose from HWC to CHW
    x = np.transpose(x, (2, 0, 1))

    # Add batch dimension
    x = np.expand_dims(x, axis=0)

    # Ensure float32
    x = x.astype(np.float32)

    return x


def predict(url):
    """
    Main prediction function.

    Args:
        url: URL of the image to classify

    Returns:
        Prediction score (float)
    """
    # Download and prepare image
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))

    # Preprocess
    x = preprocess_input(img)

    # Run inference
    result = session.run([output_name], {input_name: x})
    output = result[0][0][0]

    return float(output)


def lambda_handler(event, context):
    """
    AWS Lambda handler function.

    Expected event format:
    {
        "url": "https://example.com/image.jpg"
    }

    Returns:
    {
        "prediction": 0.09
    }
    """
    url = event['url']
    prediction = predict(url)

    return {
        'prediction': prediction
    }
