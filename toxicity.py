import os
import pickle
import pandas as pd
import tensorflow as tf
from keras.layers import TextVectorization
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# Load the dataset
df = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
)

X = df["comment_text"]
y = df[df.columns[2:]].values

MAX_FEATURES = 200000  # number of words in the vocab

# Text vectorization
vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode="int"
)

vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

# Save the vectorizer
pickle.dump(
    {"config": vectorizer.get_config(), "weights": vectorizer.get_weights()},
    open("tv_layer.pkl", "wb"),
)

# Create a dataset
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)  # helps bottlenecks

train = dataset.take(int(len(dataset) * 0.7))
val = dataset.skip(int(len(dataset) * 0.7)).take(int(len(dataset) * 0.2))
test = dataset.skip(int(len(dataset) * 0.9)).take(int(len(dataset) * 0.1))

# Model architecture

model = Sequential()
model.add(Embedding(MAX_FEATURES + 1, 100))  # Example with embedding dimension 100
model.add(Bidirectional(LSTM(32, activation="tanh")))
model.add(Dropout(0.3))  # Example dropout layer after LSTM
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))  # Example dropout layer after dense layer
model.add(Dense(256, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())  # Batch normalization layer
model.add(Dense(128, activation="relu"))
model.add(Dense(6, activation="sigmoid"))

# Compile the model with additional metrics
model.compile(
    loss="BinaryCrossentropy",
    optimizer="Adam",
    metrics=["accuracy"],
)

model.summary()

# Set up ModelCheckpoint callback to save the best model based on validation loss
checkpoint_filepath = "toxicity_model_checkpoint.h5"
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

# Set up ReduceLROnPlateau callback to reduce learning rate on plateau
reduce_lr_callback = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1,
)

# Set up EarlyStopping callback to stop training early if validation loss doesn't improve
early_stopping_callback = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=13,
    restore_best_weights=True,
    verbose=1,
)

# Train the model with callbacks
history = model.fit(
    dataset,
    epochs=35,
    validation_data=val,
    callbacks=[model_checkpoint_callback, reduce_lr_callback, early_stopping_callback],
)

# Save the final model
model.save("toxicity_model_final.h5")
