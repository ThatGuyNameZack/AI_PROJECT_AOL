import cv2
import numpy as np
import logging

def preprocess_frame(frame):
    """
    Preprocess frame for emotion detection
    
    Args:
        frame (numpy.ndarray): Input image frame
    
    Returns:
        numpy.ndarray: Preprocessed frame or None if preprocessing fails
    """
    try:
        # Check for empty or invalid frame
        if frame is None or frame.size == 0:
            logging.warning("Empty or invalid frame received")
            return None
        
        # Ensure the frame is in color
        if len(frame.shape) < 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Resize and normalize
        frame_resized = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
        frame_normalized = frame_resized / 255.0
        frame_preprocessed = frame_normalized.reshape(1, 64, 64, 3)
        
        return frame_preprocessed
    
    except Exception as e:
        logging.error(f"Error in preprocess_frame: {e}")
        return None