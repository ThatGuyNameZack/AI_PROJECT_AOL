import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
from config import EMOTION_LABELS, COLOR_MAP

def draw_emotion_info(frame, x, y, w, h, emotion, confidence, color):
    """
    Draw emotion information on the frame
    
    Args:
        frame (numpy.ndarray): Input frame
        x, y, w, h (int): Face rectangle coordinates
        emotion (str): Detected emotion
        confidence (float): Emotion confidence
        color (tuple): Color for drawing
    
    Returns:
        numpy.ndarray: Frame with drawn emotion info
    """
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Display emotion text
    display_text = f"{emotion} ({confidence:.2f})"
    cv2.putText(frame, display_text, (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                color, 2)
    
    return frame

def plot_emotion_confidences(emotion_avg_confidences):
    """
    Create a bar graph of emotion confidences
    
    Args:
        emotion_avg_confidences (list): Average confidences for each emotion
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(EMOTION_LABELS, emotion_avg_confidences, 
                color=['green', 'orange', 'red', 'gray', 'magenta', 'yellow'])
        plt.title('Emotion Detection - Average Confidences', fontsize=15)
        plt.xlabel('Emotions', fontsize=12)
        plt.ylabel('Average Confidence', fontsize=12)
        plt.ylim(0, 1)

        # Add value labels on top of each bar
        for i, v in enumerate(emotion_avg_confidences):
            plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error creating bar graph: {e}")