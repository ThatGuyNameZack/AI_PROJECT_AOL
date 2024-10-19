import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model (change the path as needed)
model = load_model('path/to/your/model.h5')

# Define a function to preprocess the image for the model
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to the input size of your model
    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Make predictions
    predictions = model.predict(processed_frame)
    brand_index = np.argmax(predictions)  # Get the index of the highest probability
    # You can map brand_index to brand names if you have a list of brands
    brand_name = "Brand {}".format(brand_index)  # Replace with your brand names

    # Display the brand name on the frame
    cv2.putText(frame, brand_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame with the detected brand
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
