# 8.2
# TensorFlow and Keras
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img

path = './clothing-dataset-small/train/t-shirt/'
name = '5f0a3fa0-6a3d-4b68-b213-72766a643de7.jpg'
fullname = f'{path}{name}'

img = load_img(fullname, target_size=(299, 299))
print(img)
x = np.array(img)
print(x.shape)

# 8.3
# Pre-trained convolutional neural networks
# - imagenet dataset: https://www.image-net.org/
# - pre-trained models: https://keras.io/api/applications/

from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions

model = Xception(weights='imagenet', input_shape=(299, 299, 3))

X = np.array([x, x, x])
print(X.shape)
X = preprocess_input(X)
pred = model.predict(X)
print(decode_predictions(pred))
print(pred.shape)
print(X[0])

# 8.4
# Convolutional Neural Networks
# - Types of layers: convolutional and dense
# - Convolutional layers and filters
# - Dense layers
# Theory


# 8.5
# Transfer Learning
# - Reading data with ImageDataGenerator
# - Train Xception on smaller images (150x150)
# Better to run it with GPU

img150 = load_img(fullname, target_size=(150, 150))
x150 = np.array(img150)
X150 = np.stack([x150, x150, x150], axis=0)  # (3, 150, 150, 3)
X = preprocess_input(X150)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_ds = train_gen.flow_from_directory(
    './clothing-dataset-small/train/',
    target_size=(150, 150),
    batch_size=32
)

print(train_ds.class_indices)

x, y = next(train_ds)
print(x.shape)
print(y.shape)
print(y[:5])

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_gen.flow_from_directory(
    './clothing-dataset-small/validation/',
    target_size=(150, 150),
    batch_size=32
)

base_model = keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

base_model.trainable = False

inputs = keras.Input(shape=(150, 150, 3))

base = base_model(inputs, training=False)

vectors = keras.layers.GlobalAveragePooling2D()(base)

outputs = keras.layers.Dense(10)(vectors)

model = keras.Model(inputs=inputs, outputs=outputs)

preds = model.predict(x)

print(preds[0])
print(preds.shape)

learning_rate = 0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

loss = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=val_ds)

print(history.history['accuracy'])
print(history.history['val_accuracy'])

plt.plot(history.history['accuracy'], label='train')
plt.legend()
plt.show()

plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.xticks(np.arange(10))
plt.show()


# 8.6
# Adjust the learning rate
# - What's the learning rate
# - Trying different values

def make_model(learning_rate=0.01):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #######################################################
    # could be in separate function create_architectrue

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(10)(vectors)
    model = keras.Model(inputs=inputs, outputs=outputs)

    #######################################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


scores = {}

for lr in [0.0001, 0.001, 0.01, 0.1]:
    print(lr)

    model = make_model(learning_rate=lr)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[lr] = history.history

    print()
    print()

# plot results

for lr, hist in scores.items():
    plt.plot(hist['accuracy'], label=lr)

plt.xticks(np.arange(10))
plt.legend()
plt.show()

del scores[0.0001]
del scores[0.1]

for lr, hist in scores.items():
    plt.plot(hist['accuracy'], label=f"train={lr}")
    plt.plot(hist['val_accuracy'], label=f"val={lr}")

plt.xticks(np.arange(10))
plt.legend()
plt.show()

learning_rate = 0.001

# 8.7
# Checkpointing
# - Saving the best model only
# - Training a model with callbacks

model.save_weights('model_v1.h5', save_format='h5')
'xception_v1_{epoch:02d}-{val_accuracy:.3f}.h5'.format(epoch=3, val_accuracy=0.95)

checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v1_{epoch:02d}-{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
)

learning_rate = 0.001

model = make_model(learning_rate=learning_rate)

history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[checkpoint])


# 8.8
# Adding more layers

def make_model(learning_rate=0.01, size_inner=100):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #######################################################
    # could be in separate function create_architectrue

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)

    outputs = keras.layers.Dense(10)(inner)
    model = keras.Model(inputs=inputs, outputs=outputs)

    #######################################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


learning_rate = 0.001

scores = {}

for size in [10, 100, 1000]:
    print(size)

    model = make_model(learning_rate=learning_rate, size_inner=size)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[size] = history.history

    print()
    print()

# plot results

for size, hist in scores.items():
    plt.plot(hist['accuracy'], label=f"train={size}")
    plt.plot(hist['val_accuracy'], label=f"val={size}")

plt.xticks(np.arange(10))
plt.legend()
plt.show()


# 8.9
# Regularization and dropout
# - Regularizing by freezing a part of the network
# - Adding dropout to our model
# - Experimenting with different values


def make_model(learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #######################################################
    # could be in separate function create_architectrue

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)

    outputs = keras.layers.Dense(10)(drop)
    model = keras.Model(inputs=inputs, outputs=outputs)

    #######################################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


learning_rate = 0.001
size = 100

scores = {}

for droprate in [0.0, 0.2, 0.5, 0.8]:
    print(droprate)

    model = make_model(learning_rate=learning_rate, size_inner=size, droprate=droprate)
    history = model.fit(train_ds, epochs=10, validation_data=val_ds)
    scores[size] = history.history

    print()
    print()

# plot results

for droprate, hist in scores.items():
    plt.plot(hist['accuracy'], label=f"train={droprate}")
    plt.plot(hist['val_accuracy'], label=f"val={droprate}")

plt.xticks(np.arange(10))
plt.legend()
plt.show()

droprate = 0.2

# 8.10
# Data Augmentation

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # rotation_range=30,
    # width_shift_range=10.0,
    # height_shift_range=10.0,
    shear_range=10.0,
    zoom_range=0.1,
    vertical_flip=False,
)

train_ds = train_gen.flow_from_directory(
    './clothing-dataset-small/train/',
    target_size=(150, 150),
    batch_size=32
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    './clothing-dataset-small/validation/',
    target_size=(150, 150),
    batch_size=32,
    shuffle=False
)

learning_rate = 0.001
size = 100
droprate = 0.2

model = make_model(learning_rate=learning_rate, size_inner=size, droprate=droprate)

history = model.fit(train_ds, epochs=50, validation_data=val_ds)

hist = history.history
plt.plot(hist['val_accuracy'], label='val')
plt.plot(hist['accuracy'], label='train')
plt.legend()
plt.show()


# 8.11
# Training a larger model
# - Train a 299x299 model

def make_model(input_size=150, learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #######################################################
    # could be in separate function create_architectrue

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)

    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)

    outputs = keras.layers.Dense(10)(drop)
    model = keras.Model(inputs=inputs, outputs=outputs)

    #######################################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


learning_rate = 0.001
size = 100
droprate = 0.2
input_size = 299

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=10.0,
    zoom_range=0.1,
    horizontal_flip=True,
)

train_ds = train_gen.flow_from_directory(
    './clothing-dataset-small/train/',
    target_size=(input_size, input_size),
    batch_size=32
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = val_gen.flow_from_directory(
    './clothing-dataset-small/validation/',
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v4_{epoch:02d}-{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
)

model = make_model(input_size=input_size, learning_rate=learning_rate, size_inner=size, droprate=droprate)

history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=[checkpoint])

hist = history.history
plt.plot(hist['val_accuracy'], label='val')
plt.plot(hist['accuracy'], label='train')
plt.legend()
plt.show()

# 8.12
# Using the model
# - Loading the model
# - Evaluating the model
# - Getting predictions

import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('xception_v4_13_0.903.h5')

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.xception import preprocess_input

test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_ds = test_gen.flow_from_directory(
    './clothing-dataset-small/test/',
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)

model.evaluate(test_ds)

path = './clothing-dataset-small/test/pants/c8d21106-bbdb-4e8d-83e4-bf3d14e54c16.jpg'
img = load_img(path, target_size=(299, 299))

import numpy as np

x = np.array(img)
X = np.array([x])
X.shape

X = preprocess_input(X)
pred = model.predict(X)

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

print(pred[0])

print(dict(zip(classes, pred[0])))
