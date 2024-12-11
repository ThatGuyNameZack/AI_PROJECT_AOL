import logging
import numpy as np
from emotion_detector import EmotionDetector
from visualization import plot_emotion_confidences
from config import EMOTION_LABELS

# Set up logging
logging.basicConfig(level=logging.INFO)

def run_emotion_detection():
    """
    Run emotion detection and return the average confidence for each emotion.
    """
    # Initialize and run emotion detector
    detector = EmotionDetector()
    emotion_confidences = detector.run()
    
    # Calculate average confidences
    emotion_avg_confidences = []
    
    for emotion, confidences in emotion_confidences.items():
        avg_conf = np.nanmean(confidences) if confidences else 0
        logging.info(f"{emotion}: Average Confidence = {avg_conf:.2f}")
        emotion_avg_confidences.append(avg_conf)
    
    # Plot emotion confidences
    plot_emotion_confidences(emotion_avg_confidences)
    
    return emotion_avg_confidences
