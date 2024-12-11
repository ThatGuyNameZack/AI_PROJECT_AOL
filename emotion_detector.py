import cv2
import numpy as np
import logging
import sys
import gc

from config import (
    EMOTION_LABELS, COLOR_MAP, FACE_CASCADE_PATH, 
    MAX_CONFIDENCE_HISTORY, MIN_CONFIDENCE_THRESHOLD
)
from preprocessing import preprocess_frame
from visualization import draw_emotion_info
from engage import predict_emotion

class EmotionDetector:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler('emotion_detection.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        
        # Initialize emotion confidence tracking
        self.emotion_confidences = {label: [] for label in EMOTION_LABELS}
        
        # Initialize camera
        self.cam = None
    
    def initialize_camera(self, width=1280, height=720, fps=30):
        """
        Initialize the camera with specified settings
        
        Args:
            width (int): Camera frame width
            height (int): Camera frame height
            fps (int): Frames per second
        
        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        self.cam = cv2.VideoCapture(0)
        if not self.cam.isOpened():
            logging.error("Could not open camera.")
            return False
        
        # Set camera properties
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cam.set(cv2.CAP_PROP_FPS, fps)
        self.cam.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        self.cam.set(cv2.CAP_PROP_CONTRAST, 0.5)
        
        return True
    
    def detect_emotions(self, frame):
        """
        Detect emotions in the given frame
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            tuple: Final emotion and its confidence
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        final_emotion = None
        final_confidence = 0
        
        for (x, y, w, h) in faces:
            # Extract face with padding
            padding = max(w, h) // 4
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
                emotion_probs = prediction[0]
                emotion_index = np.argmax(emotion_probs)
                confidence = emotion_probs[emotion_index]
                
                # Track predictions above threshold
                if confidence >= MIN_CONFIDENCE_THRESHOLD:
                    current_emotion = EMOTION_LABELS[emotion_index]
                    self.emotion_confidences[current_emotion].append(confidence)
                    
                    # Trim confidence history
                    for emotion in self.emotion_confidences:
                        if len(self.emotion_confidences[emotion]) > MAX_CONFIDENCE_HISTORY:
                            self.emotion_confidences[emotion] = self.emotion_confidences[emotion][-MAX_CONFIDENCE_HISTORY:]
                
                # Compute average confidences
                avg_confidences = {
                    emotion: (np.mean(confidences) if confidences else 0) 
                    for emotion, confidences in self.emotion_confidences.items()
                }
                
                # Determine final emotion
                final_emotion = max(avg_confidences, key=avg_confidences.get)
                final_confidence = avg_confidences[final_emotion]
                
                # Draw emotion info
                frame = draw_emotion_info(
                    frame, x, y, w, h, 
                    final_emotion, final_confidence, 
                    COLOR_MAP.get(final_emotion, (255, 255, 255))
                )
            
            except Exception as e:
                logging.error(f"Error during emotion prediction: {e}")
        
        return frame, (final_emotion, final_confidence)
    
    def run(self):
        """
        Main emotion detection loop
        """
        if not self.initialize_camera():
            return
        
        try:
            while True:
                # Read frame
                ret, frame = self.cam.read()
                if not ret:
                    logging.error("Could not read frame.")
                    break
                
                # Detect and visualize emotions
                frame, (emotion, confidence) = self.detect_emotions(frame)
                
                # Display the frame
                cv2.imshow('Real-Time Emotion Detection', frame)
                
                # Break loop on 'p' key press
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    break
        
        except KeyboardInterrupt:
            logging.info("\nExiting program...")
        
        finally:
            # Release resources
            if self.cam:
                self.cam.release()
            cv2.destroyAllWindows()
            gc.collect()
            
        return self.emotion_confidences
