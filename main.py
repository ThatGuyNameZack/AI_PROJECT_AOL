import site
import sys
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from engage import predict_emotion
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s',
                    handlers=[
                        logging.FileHandler('emotion_detection.log'),
                        logging.StreamHandler(sys.stdout)
                    ])

# Initialize webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    logging.error("Could not open camera.")
    sys.exit(1)

# Load Haar Cascade
base_dir = os.path.dirname(os.path.realpath(__file__))
face_cascade_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
if not os.path.exists(face_cascade_path):
    logging.error(f"Haar cascade file not found at {face_cascade_path}.")
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(face_cascade_path)
threshold = 0.3 #to detect the faces

# Emotion labels with careful ordering
emotion_labels = ['Engaged', 'Confused', 'Frustrated', 'Bored', 'Drowsy', 'Distracted']

# Preprocessing function with enhanced error handling
def preprocess_frame(frame):
    try:
        # More robust color conversion and resizing
        if frame is None or frame.size == 0:
            logging.warning("Empty or invalid frame received")
            return None
        
        # Ensure the frame is in color and resize
        if len(frame.shape) < 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # More robust resizing with interpolation
        frame_resized = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        frame_resized = frame_resized / 255.0  # Normalize pixel values
        frame_resized = frame_resized.reshape(1, 64, 64, 3)  # Add batch dimension
        return frame_resized
    except Exception as e:
        logging.error(f"Error in preprocess_frame: {e}")
        return None

# Emotion detection confidence tracking with more robust handling
emotion_confidences = {label: [] for label in emotion_labels}
MAX_CONFIDENCE_HISTORY = 30  # Increased for more stable tracking
MIN_CONFIDENCE_THRESHOLD = 0.1  # Ignore very low confidence predictions

try:
    # Main loop
    while True:
        ret, frame = cam.read()
        if not ret:
            logging.error("Could not read frame.")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                               minNeighbors=5, 
                                               minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract face ROI with padding
            padding = 10
            padded_x = max(0, x - padding)
            padded_y = max(0, y - padding)
            padded_w = min(frame.shape[1] - padded_x, w + 2 * padding)
            padded_h = min(frame.shape[0] - padded_y, h + 2 * padding)
            face = frame[padded_y:padded_y + padded_h, padded_x:padded_x + padded_w]
            
            # Preprocess the face
            processed_face = preprocess_frame(face)
            
            if processed_face is None:
                continue

            try:
                # Predict emotion
                prediction = predict_emotion(processed_face)  
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                continue

            # Process prediction
            emotion_probs = prediction[0]
            emotion_index = np.argmax(emotion_probs)
            confidence = emotion_probs[emotion_index]
            
            # Only track predictions above threshold
            if confidence >= MIN_CONFIDENCE_THRESHOLD:
                # Update confidence history
                current_emotion = emotion_labels[emotion_index]
                emotion_confidences[current_emotion].append(confidence)
                
                # Trim confidence history
                for emotion in emotion_confidences:
                    if len(emotion_confidences[emotion]) > MAX_CONFIDENCE_HISTORY:
                        emotion_confidences[emotion] = emotion_confidences[emotion][-MAX_CONFIDENCE_HISTORY:]
            
            # Compute average confidence for each emotion with fallback
            avg_confidences = {
                emotion: (np.mean(confidences) if confidences else 0) 
                for emotion, confidences in emotion_confidences.items()
            }
            
            # Determine final emotion with highest average confidence
            final_emotion = max(avg_confidences, key=avg_confidences.get)
            final_confidence = avg_confidences[final_emotion]

            # Detailed logging instead of printing
            logging.info("\n--- Emotion Prediction ---")
            for label, prob in zip(emotion_labels, emotion_probs):
                logging.info(f"{label}: {prob:.2f}")
            logging.info(f"\nCurrent Emotion: {final_emotion}")
            logging.info(f"Confidence: {final_confidence:.2f}")
            logging.info("Average Confidences: %s", 
                  {emotion: f"{conf:.2f}" for emotion, conf in avg_confidences.items()})

            # Draw rectangle around face and display emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            
            # Set camera resolution
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height

            # Adjust other properties
            cam.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Brightness (0.0 to 1.0)
            cam.set(cv2.CAP_PROP_CONTRAST, 0.5)    # Contrast (0.0 to 1.0)
            
            # Color code the text based on emotion
            color_map = {
                'Engaged': (0, 255, 0),      # Green
                'Confused': (255, 165, 0),   # Orange
                'Frustrated': (0, 0, 255),   # Red
                'Bored': (128, 128, 128),    # Gray
                'Drowsy': (255, 0, 255),     # Magenta
                'Distracted': (255, 255, 0) # Yellow
            }
            
            # Use final emotion for display
            display_text = f"{final_emotion} ({final_confidence:.2f})"
            cv2.putText(frame, display_text, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                        color_map.get(final_emotion, (255, 255, 255)), 2)

        # Display the frame
        cv2.imshow('Real-Time Emotion Detection', frame)

        # Break loop on 'p' key press
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break

except KeyboardInterrupt:
    logging.info("\nExiting program...")

finally:
    # Release resources
    cam.release()
    cv2.destroyAllWindows()

# Print final emotion statistics with more robust handling
logging.info("\n--- Emotion Detection Summary ---")
emotion_avg_confidences = []
for emotion, confidences in emotion_confidences.items():
    # Use numpy's nanmean to handle potential empty lists
    avg_conf = np.nanmean(confidences) if confidences else 0
    logging.info(f"{emotion}: Average Confidence = {avg_conf:.2f}")
    emotion_avg_confidences.append(avg_conf)

# Create bar graph with error handling
try:
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_labels, emotion_avg_confidences, 
            color=['green', 'orange', 'red', 'gray', 'magenta', 'yellow'])
    plt.title('Emotion Detection - Average Confidences', fontsize=15)
    plt.xlabel('Emotions', fontsize=12)
    plt.ylabel('Average Confidence', fontsize=12)
    plt.ylim(0, 1)  # Set y-axis from 0 to 1

    # Add value labels on top of each bar
    for i, v in enumerate(emotion_avg_confidences):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()
except Exception as e:
    logging.error(f"Error creating bar graph: {e}")