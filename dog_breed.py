import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

labels_csv = pd.read_csv("drive/My Drive/Dog Vision/labels.csv")
# labels_csv.describe()
# print(labels_csv.head())


filenames = [
    "drive/My Drive/Dog Vision/train/" + fname + ".jpg" for fname in labels_csv["id"]
]


import os

if len(os.listdir("drive/My Drive/Dog Vision/train/")) == len(filenames):
    print("Filenames match actual amount of files!!! Proceed.")
else:
    print("Filenames do not match actual amount of files, check the target directory.")


import numpy as np

labels = labels_csv["breed"].to_numpy()  # convert labels column to NumPy array
labels[:10]


unique_breeds = np.unique(labels)
len(unique_breeds)


boolean_labels = [label == np.array(unique_breeds) for label in labels]
boolean_labels[:2]


X = filenames
y = boolean_labels


NUM_IMAGES = 1000

from sklearn.model_selection import train_test_split

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X[:NUM_IMAGES], y[:NUM_IMAGES], test_size=0.2, random_state=42
)
len(X_train), len(X_val), len(y_train), len(y_val)

## Preprocessing Images (turning images into Tensors)
from matplotlib.pyplot import imread

image = imread(filenames[42])
image.shape


IMG_SIZE = 224


def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image


def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label


# Define the batch size, 32 is a good default
BATCH_SIZE = 32


# Create a function to turn data into batches
def create_data_batches(
    x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False
):
    """
    Creates batches of data out of image (x) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
    Also accepts test data as input (no labels).
    """
    # If the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))  # only filepaths
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    # If the data if a valid dataset, we don't need to shuffle it
    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(x), tf.constant(y))  # filepaths
        )  # labels
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    else:
        # If the data is a training dataset, we shuffle it
        print("Creating training data batches...")
        # Turn filepaths and labels into Tensors
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(x), tf.constant(y))  # filepaths
        )  # labels

        # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
        data = data.shuffle(buffer_size=len(x))

        # Create (image, label) tuples (this also turns the image path into a preprocessed image)
        data = data.map(get_image_label)

        # Turn the data into batches
        data_batch = data.batch(BATCH_SIZE)
    return data_batch


train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)


# Visualize
import matplotlib.pyplot as plt


def show_25_images(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(unique_breeds[labels[i].argmax()])


train_images, train_labels = next(train_data.as_numpy_iterator())
train_images, train_labels


# Model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]  # batch, height, width, color channels
OUTPUT_SHAPE = len(unique_breeds)
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


# Create a function which builds a Keras model
def create_model(
    input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL
):
    print("Building model with:", model_url)

    model = tf.keras.Sequential(
        [
            hub.KerasLayer(model_url),
            tf.keras.layers.Dense(units=output_shape, activation="softmax"),
        ]
    )

    # Compile
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    model.build(input_shape)
    return model


model = create_model()
model.summary()


import datetime


# TensorBoard callback function
def create_tensorboard_callback():
    logdir = os.path.join(
        "drive/My Drive/Dog Vision/logs",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    return tf.keras.callbacks.TensorBoard(logdir)


early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)


NUM_EPOCHS = 100


def train_model():
    """
    Trains a given model and returns the trained version.
    """
    # Create a model
    model = create_model()

    # Create new TensorBoard session everytime we train a model
    tensorboard = create_tensorboard_callback()

    # Fit the model to the data passing it the callbacks we created
    model.fit(
        x=train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
        validation_freq=1,  # check validation metrics every epoch
        callbacks=[tensorboard, early_stopping],
    )

    return model


model = train_model()


predictions = model.predict(val_data, verbose=1)
predictions


full_data = create_data_batches(X, y)
full_model = create_model()


full_model_tensorboard = create_tensorboard_callback()

# Early stopping callback
# Note: No validation set when training on all the data, therefore can't monitor validation accruacy
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy", patience=3
)

# Fit the full model to the full training data
full_model.fit(
    x=full_data,
    epochs=NUM_EPOCHS,
    callbacks=[full_model_tensorboard, full_model_early_stopping],
)
