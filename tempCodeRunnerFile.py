from collections import deque
from flask import Flask, render_template, Response
import io
import base64
import logging
import matplotlib
matplotlib.use('Agg')  # Add this before importing pyplot
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from config import (
    EMOTION_LABELS, COLOR_MAP, FACE_CASCADE_PATH, 
    MAX_CONFIDENCE_HISTORY, MIN_CONFIDENCE_THRESHOLD,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
)
from preprocessing import preprocess_frame
from engage import predict_emotion
from visualization import plot_emotion_confidences  # Ensure that you have this method

app = Flask(__name__)
confidence_history = deque(maxlen=MAX_CONFIDENCE_HISTORY)

# Initialize the camera
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def draw_emotion_info(frame, x, y, w, h, emotion, confidence, color):
    """
    Draw bounding box and emotion information on the frame.
    """
    try:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Display emotion and confidence
        display_text = f"{emotion} ({confidence:.2f})"
        cv2.putText(
            frame, display_text, (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    except Exception as e:
        logging.error(f"Error in draw_emotion_info: {e}")
    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                # Extract and preprocess face
                face = frame[y:y + h, x:x + w]
                processed_face = preprocess_frame(face)
                
                if processed_face is not None:
                    # Predict emotion
                    prediction = predict_emotion(processed_face)
                    emotion_probs = prediction[0]
                    
                    # Append confidences to global history
                    confidence_history.append(emotion_probs)

                    # Find the most confident emotion
                    emotion_index = np.argmax(emotion_probs)
                    emotion = EMOTION_LABELS[emotion_index]
                    confidence = emotion_probs[emotion_index]

                    # Display emotion if confidence is above threshold
                    if confidence >= MIN_CONFIDENCE_THRESHOLD:
                        color = COLOR_MAP.get(emotion, (255, 255, 255))
                        frame = draw_emotion_info(frame, x, y, w, h, emotion, confidence, color)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame as part of a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Main page with a link to emotion summary."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for live video feed."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )

@app.route('/emotion_summary')
def emotion_summary():
    """
    Generate and display emotion detection summary.
    """
    try:
        # Calculate average confidences from history
        if len(confidence_history) > 0:
            emotion_avg_confidences = np.mean(confidence_history, axis=0)
        else:
            # Default to zeros if no history is available
            emotion_avg_confidences = [0.0] * len(EMOTION_LABELS)

        # Create confidences dictionary for template
        confidences = dict(zip(EMOTION_LABELS, emotion_avg_confidences))

        # Generate the plot
        plot_img_base64 = plot_emotion_confidences(emotion_avg_confidences)

        # Render the template with plot and confidences
        return render_template('emotion_summary.html', 
                               confidences=confidences, 
                               plot_img=plot_img_base64)

    except Exception as e:
        logging.error(f"Emotion summary error: {e}")
        return render_template('emotion_summary.html', 
                               error_message="Emotion summary feature is still in development.")

# Run Flask app
if __name__ == '__main__':
    app.run(port=5001, debug=True, threaded=True)



