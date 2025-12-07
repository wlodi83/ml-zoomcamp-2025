PREFIX="https://github.com/alexeygrigorev/large-datasets/releases/download/hairstyle"
DATA_URL="${PREFIX}/hair_classifier_v1.onnx.data"
MODEL_URL="${PREFIX}/hair_classifier_v1.onnx"
wget ${DATA_URL}
wget ${MODEL_URL}
