import logging
import numpy as np

from emotion_detector import EmotionDetector
from visualization import plot_emotion_confidences
from config import EMOTION_LABELS

def main():
    # Initialize and run emotion detector
    detector = EmotionDetector()
    emotion_confidences = detector.run()
    
    # Calculate and log average confidences
    logging.info("\n--- Emotion Detection Summary ---")
    emotion_avg_confidences = []
    
    for emotion, confidences in emotion_confidences.items():
        avg_conf = np.nanmean(confidences) if confidences else 0
        logging.info(f"{emotion}: Average Confidence = {avg_conf:.2f}")
        emotion_avg_confidences.append(avg_conf)
    
    # Plot emotion confidences
    plot_emotion_confidences(emotion_avg_confidences)

if __name__ == "__main__":
    main()