from flask import Flask, render_template, Response
import cv2
import numpy as np
from config import (
    EMOTION_LABELS, COLOR_MAP, FACE_CASCADE_PATH, 
    MAX_CONFIDENCE_HISTORY, MIN_CONFIDENCE_THRESHOLD,
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
)
from preprocessing import preprocess_frame
from engage import predict_emotion  # Ensure that you have this method

app = Flask(__name__)

# Initialize the camera
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Extract face for emotion detection
                face = frame[y:y + h, x:x + w]
                processed_face = preprocess_frame(face)
                
                if processed_face is not None:
                    # Predict emotion
                    prediction = predict_emotion(processed_face)
                    emotion_probs = prediction[0]
                    emotion_index = np.argmax(emotion_probs)
                    confidence = emotion_probs[emotion_index]
                    
                    # Display emotion if confidence is above threshold
                    if confidence >= MIN_CONFIDENCE_THRESHOLD:
                        emotion = EMOTION_LABELS[emotion_index]
                        color = COLOR_MAP.get(emotion, (255, 255, 255))
                        
                        # Draw emotion label on the frame
                        cv2.putText(frame, f'{emotion}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

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
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/emotion_summary')
def emotion_summary():
    """Placeholder for emotion summary processing."""
    return "<h1>Emotion summary feature is under development.</h1>"

# Run Flask app
if __name__ == '__main__':
    app.run(port=5001, debug=True, threaded=True)


