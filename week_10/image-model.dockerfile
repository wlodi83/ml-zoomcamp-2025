FROM tensorflow/serving:2.7.0

COPY clothing-model /models/cloting-model/1
ENV MODEL_NAME="cloting-model"