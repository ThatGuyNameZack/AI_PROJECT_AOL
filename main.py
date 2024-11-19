import site
import sys
import cv2
import numpy as np
import os  # Cross-platform path handling
from engage import predict_emotion

print(site.getsitepackages())  # Debugging for library paths

cam = cv2.VideoCapture(0)
print("Tracking module imported.")

if not cam.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

# Path for Haar Cascade
base_dir = os.path.dirname(os.path.realpath(__file__))
face_cascade_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

threshold = 0.7  # Confidence threshold

def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize and normalize
    frame_resized = cv2.resize(frame, (64, 64))
    frame_resized = frame_resized / 255.0
    frame_resized = frame_resized.reshape(1, 64, 64, 3)
    return frame_resized

# Main loop
while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Extract face ROI
        processed_face = preprocess_frame(face)
        prediction = predict_emotion(processed_face)
        emotion = np.argmax(prediction)
        confidence = prediction[0][emotion]

        emotion_labels = ['Engaged', 'Confused', 'Frustrated', 'Bored', 'Drowsy', 'Looking Away']
        emotion_text = emotion_labels[emotion] if confidence >= threshold else "Not Engaged"

        print(f"RAW PREDICTIONS: {prediction}")
        print(f"Predicted emotion: {emotion_text} with confidence {confidence}")

        # Draw rectangle around face and display emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('camera', frame)

    if cv2.waitKey(1) == ord("p"):
        break

cam.release()
cv2.destroyAllWindows()
 
