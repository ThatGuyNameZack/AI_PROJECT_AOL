import cv2
import tensorflow as tf
import numpy as np
import config

# Load the trained model
model = tf.keras.models.load_model('emotion_model.h5')

# Class names corresponding to the model output
class_names = ['engaged', 'bored', 'drowsy', 'frustrated', 'looking away', 'confused']

def TrackingObject():
    # Open video capture
    video_capture = cv2.VideoCapture(0)
    tracker = cv2.TrackerKCF_create()

    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        return

    # Manually select ROI (Region of Interest) for face detection
    bbox = cv2.selectROI("Select face", frame, fromCenter=False, showCrosshair=True)
    tracker.init(frame, bbox)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert the frame to grayscale (for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load Haar Cascade classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # If faces are detected, predict emotion
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

            # Crop the face from the frame and resize to match model input size
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (64, 64))  # Resize to 64x64 (match model input)
            face_normalized = face_resized / 255.0  # Normalize pixel values
            face_expanded = np.expand_dims(face_normalized, axis=0)  # Add batch dimension

            # Make the prediction for the face
            prediction = model.predict(face_expanded)

            # Get the class with the highest probability
            predicted_class = np.argmax(prediction[0])
            predicted_confidence = prediction[0][predicted_class]

            # Display the class only if the confidence is above a certain threshold (e.g., 0.5)
            if predicted_confidence > 0.5:  # Confidence threshold
                class_name = class_names[predicted_class]
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Uncertain", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Update the tracker
        success, bbox = tracker.update(frame)
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

        # Display the frame with predictions and bounding boxes
        cv2.imshow("Tracking", frame)

        # Break the loop if the user presses 'p'
        if cv2.waitKey(1) == ord("p"):
            break

    # Release the camera and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TrackingObject()
