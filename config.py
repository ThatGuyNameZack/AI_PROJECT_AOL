import main 
import matplotlib
import cv2
import os

#this will be used for data ser pathawy

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset directory and file paths
DATASET_DIR = os.path.join(BASE_DIR, 'Student-engagement-dataset')

# Paths to the 'engaged' and 'not engaged' subfolders
ENGAGED_DATA_PATH = os.path.join(DATASET_DIR, 'engaged')
NOT_ENGAGED_DATA_PATH = os.path.join(DATASET_DIR, 'not engaged')

# Face detection model parameters
FACE_DETECTION_MODEL = os.path.join(BASE_DIR, 'models', 'haarcascade_frontalface_default.xml')

# Dataset-specific parameters (example)
IMG_SIZE = (224, 224)  # Target image size for resizing
BATCH_SIZE = 32        # Batch size for loading data
NUM_CLASSES = 2        # Number of classes in your dataset (engaged, not engaged)

# Other relevant configurations
THRESHOLD = 0.5        # Confidence threshold for detections, if applicable


