import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Define paths to dataset
base_dir = '/Users/rafisatria/Documents/GitHub/AI_PROJECT_AOL/Student-engagement-dataset/'
engaged_dir = os.path.join(base_dir, 'engaged')
not_engaged_dir = os.path.join(base_dir, 'not engaged')

# Image dimensions
img_width, img_height = 64, 64

# Load images and labels
def load_images_and_labels():
    images = []
    labels = []
    
    for label, folder in zip(['Engaged', 'Confused', 'Frustrated', 'Bored', 'Drowsy', 'Looking Away'],
                             [engaged_dir, engaged_dir, engaged_dir, not_engaged_dir, not_engaged_dir, not_engaged_dir]):
        for file in os.listdir(folder):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_width, img_height))
                images.append(img)
                labels.append(label)
    
    return np.array(images), np.array(labels)

# Load and preprocess the dataset
X, y = load_images_and_labels()

# Normalize images
X = X.astype('float32') / 255.0

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')  # 6 emotions
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the trained model
model.save('emotion_model.h5')

print("Model training complete and saved as emotion_model.h5")
