import cv2
from keras import layers, models
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split


# Define paths
dataset_path = 'DataSet'
model_path = 'Model/model1.h5'
results_path = 'Classification report/Classification report for model 1.json'

# Parameters
image_size = (32, 32)
batch_size = 32


# Load and preprocess dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    class_names = []

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        class_names.append(class_name)
        print(class_name)
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale mode
            if img is not None:
                img = cv2.resize(img, image_size)  # Resize image
                _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)  # Convert to binary image
                img = cv2.bitwise_not(img)  # Invert colors
                images.append(img)
                labels.append(class_name)

    return np.array(images), np.array(labels), class_names


X, y, class_names = load_images_from_folder(dataset_path)

# Normalize pixel values
X = X / 255.0
X = np.expand_dims(X, axis=-1)

# Convert class names to indices
class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
y = np.array([class_to_index[label] for label in y])
# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Class names
# class_names = train_ds.class_names
# print(class_names)
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# Define the model
model = models.Sequential()

# Input layer and first convolutional layer
model.add(layers.InputLayer(input_shape=(32, 32, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# Second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# Third convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))

# Dropout layer
model.add(layers.Dropout(0.5))

# Flatten layer
model.add(layers.Flatten())

# First dense layer
model.add(layers.Dense(512, activation='relu'))

# Dropout layer
model.add(layers.Dropout(0.5))

# Second dense layer
model.add(layers.Dense(256, activation='relu'))

# Output layer
model.add(layers.Dense(len(class_names), activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    epochs=40,
    verbose=1
)

# Save the trained model as an H5 file
model.save(model_path)

# Evaluate the model
results = model.evaluate(X_test, y_test)

# Predict on test data
predictions = model.predict(X_test)

# Create a results dictionary
results_dict = {
    'test_loss': results[0],
    'test_accuracy': results[1],
    'training set size': len(X_train),
    'Test set size:': len(X_test),
    'Validation': len(X_val),
    'number of classes': len(class_names),
    'number of epochs in training':40,
}

# Save results to a readable JSON file
with open(results_path, 'w') as json_file:
    json.dump(results_dict, json_file, indent=4)

print("Model and results have been saved.")