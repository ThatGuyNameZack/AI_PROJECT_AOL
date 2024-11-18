
import site
import sys
import cv2
import numpy as np
import os #it will work windows or unix machines
from engage import predict_emotion

print(site.getsitepackages())  # Print site packages for debugging

cam = cv2.VideoCapture(0)
print("Tracking module imported.")

if not cam.isOpened():
    print("Error: Could not open camera.")
    sys.exit()


#path set for haas tracking
base_dir  = os.path.dirname(os.path.realpath(__file__))
face_cascade_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_resized = cv2.resize(frame, (64, 64))
    frame_resized = frame_resized / 255.0
    frame_resized = frame_resized.reshape(1, 64, 64, 3)

    return frame_resized
# Main loop to read frames from the camera
while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    
   
    processed_frame = preprocess_frame(frame)
    prediction = predict_emotion(processed_frame)
    emotion = np.argmax(prediction)
    print("RAW PREDICTIONS: ", prediction)
    
    emotion_labels = ['Engaged', 'Confused', 'Frustrated', 'Bored', 'Drowsy', 'Looking Away']
    
 
    if prediction >= 0.7:
        print("Engaged")
    else:
        print("Not Engaged")

    
    if prediction[0][emotion] < threshold:
        emotion_text = "uncertain"
    else:
        emotion_text = emotion_labels[emotion]
    
    
    
    #emotion_text = emotion_labels[emotion]
    
    print(f"Predicted emotion: {emotion_text} with confidence {prediction[0][emotion]}")

    
    cv2.putText(frame, str(emotion_text), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw white box around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Display the frame in a window named 'camera'
    cv2.imshow('camera', frame)

    # Press 'p' to exit the loop and close the camera
    if cv2.waitKey(1) == ord("p"):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
