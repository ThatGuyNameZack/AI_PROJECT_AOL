import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to dataset
base_dir = '/Users/rafisatria/Documents/GitHub/AI_PROJECT_AOL/Student-engagement-dataset/'
categories = ['engaged', 'not engaged']  # Dynamically infer labels

# Image dimensions
img_width, img_height = 64, 64

# Load images and labels dynamically
def load_images_and_labels(base_dir, categories):
    images = []
    labels = []
    for category in categories:
        folder_path = os.path.join(base_dir, category)
        if not os.path.exists(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_width, img_height))
                images.append(img)
                labels.append(category)
    return np.array(images), np.array(labels)

# Load and preprocess the dataset
X, y = load_images_and_labels(base_dir, categories)

# Normalize images
X = X.astype('float32') / 255.0

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Data Augmentation
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')  # Output neurons match the number of categories
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(
    data_generator.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=20
)

# Save the trained model
model.save('emotion_model.h5')

print("Model training complete and saved as emotion_model.h5")
