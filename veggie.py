import os
import tensorflow as tf
import numpy as np
from keras import layers, models, optimizers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input


train_dir_path = "/kaggle/input/fruit-and-vegetable-image-recognition/train"
test_dir_path = "/kaggle/input/fruit-and-vegetable-image-recognition/validation"
# val_dir_path = "valid"


img_height = 224
img_width = 224
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir_path,
    labels="inferred",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=110,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir_path,
    labels="inferred",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=110,
)

class_names = train_ds.class_names

class_names_to_index = {
    class_name: index for index, class_name, in enumerate(class_names)
}

# bring 9 images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i].numpy())])
        plt.axis("off")

plt.show()


num_classes = len(class_names)

from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from keras.models import Sequential
import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Define the input shape
input_shape = (img_height, img_width, 3)

# Create the InceptionV3 base model (excluding the top layer)
base_model = tf.keras.applications.EfficientNetB2(include_top=False)
for layer in base_model.layers:
    layer.trainable = True


# Data Augmentation
data_augmentation = Sequential(
    [
        keras.layers.Rescaling(1.0 / 255),
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.05),
    ]
)

from keras.layers import BatchNormalization

model = Sequential(
    [
        data_augmentation,
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(
            1024, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        Dropout(0.3),
        BatchNormalization(),
        Dense(num_classes, activation="softmax"),
    ]
)


from keras.optimizers import Adam

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Rest of your code remains the same
from keras.callbacks import LearningRateScheduler

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,  # multiply the learning rate by 0.2 (reduce by 5x)
    patience=3,
    verbose=1,  # print out when learning rate goes down
    min_lr=1e-7,
)


# Callbacks for early stopping
callbacks = [
    EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode="min"),
    ModelCheckpoint(
        filepath="newfoodxd.h5",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
]


history = model.fit(
    train_ds,
    epochs=100,  # fine-tune for a maximum of 100 epochs
    validation_data=val_ds,
    # validation during training on 15% of test data
    callbacks=callbacks,
)

model.save("foodnihai.h5")


#####################
# TESTING
#####################

img = tf.keras.utils.load_img(
    "/kaggle/input/fruit-and-vegetable-image-recognition/test/chilli pepper/Image_4.jpg",
    target_size=(img_height, img_width),
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score)
    )
)


import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

class_names = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
    "watermelon",
]

# Load the model
loaded_model = tf.keras.models.load_model("/kaggle/input/veggie/veggie99.h5")

# Define the image dimensions
img_height, img_width = 224, 224

# Define the path to the test folder
test_folder = "/kaggle/input/fruit-and-vegetable-image-recognition/test"

# Initialize empty lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Loop through each class in the test folder
for class_name in os.listdir(test_folder):
    class_path = os.path.join(test_folder, class_name)

    # Loop through each image in the class
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Load and preprocess the image
        img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # Make predictions
        predictions = loaded_model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        # Get the true label from the folder name
        true_class = class_name

        # Append true and predicted labels to the lists
        true_labels.append(true_class)
        predicted_labels.append(class_names[predicted_class])

# Generate and print the classification report
print(classification_report(true_labels, predicted_labels))
