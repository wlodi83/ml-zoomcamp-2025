import onnxruntime as ort

onnx_model_path = "hair_classifier_v1.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name

print("Question 1:")
print("output_name: ", output_name)

# Preparing the image

from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

# Question 2: Download and resize the test image
print("\nQuestion 2:")
test_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

# Based on homework 8, the target size is 200x200
target_size = (200, 200)
print(f"Target size (from homework 8): {target_size}")

# Download and prepare the image
img = download_image(test_url)
print(f"Downloaded image size: {img.size}")

img_prepared = prepare_image(img, target_size)
print(f"Resized image size: {img_prepared.size}")
print(f"Answer: 200x200")

# Question 3: Preprocessing and first pixel value
print("\nQuestion 3:")

# Convert PIL Image to numpy array
import numpy as np
x = np.array(img_prepared, dtype='float32')
print(f"Array shape after conversion: {x.shape}")  # Should be (200, 200, 3) - HWC format

# Normalize from [0, 255] to [0, 1]
x = x / 255.0

# Apply ImageNet normalization (from homework 8)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
x = (x - mean) / std

print(f"Array shape after normalization: {x.shape}")  # Still (200, 200, 3) - HWC format

# Get the first pixel, R channel (channel 0)
first_pixel_r = x[0, 0, 0]
print(f"First pixel, R channel value: {first_pixel_r}")
print(f"Answer: {first_pixel_r:.2f}")

# Question 4: Model inference
print("\nQuestion 4:")

# ONNX model expects input in format: (batch, channels, height, width)
# Currently x is in format (height, width, channels) - HWC format
# Need to transpose to CHW format and add batch dimension

# Transpose from HWC to CHW
x_chw = np.transpose(x, (2, 0, 1))  # Now shape is (3, 200, 200)
print(f"Shape after transpose to CHW: {x_chw.shape}")

# Add batch dimension
x_batch = np.expand_dims(x_chw, axis=0)  # Now shape is (1, 3, 200, 200)

# Convert to float32 (ONNX expects float32, not float64)
x_batch = x_batch.astype(np.float32)
print(f"Shape after adding batch dimension: {x_batch.shape}")
print(f"Data type: {x_batch.dtype}")

# Run inference
result = session.run([output_name], {input_name: x_batch})
output = result[0][0][0]  # Extract the scalar output

print(f"Model output: {output}")
print(f"Answer: {output:.2f}")