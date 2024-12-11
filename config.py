import os

# Emotion labels with careful ordering
EMOTION_LABELS = ['Engaged', 'Confused', 'Frustrated', 'Bored', 'Drowsy', 'Distracted']

# Emotion color mapping
COLOR_MAP = {
    'Engaged': (0, 255, 0),      # Green
    'Confused': (255, 165, 0),   # Orange
    'Frustrated': (0, 0, 255),   # Red
    'Bored': (128, 128, 128),    # Gray
    'Drowsy': (255, 0, 255),     # Magenta
    'Distracted': (255, 255, 0)  # Yellow
}

# Cascade Classifier Path
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
FACE_CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# Detection Parameters
MAX_CONFIDENCE_HISTORY = 30
MIN_CONFIDENCE_THRESHOLD = 0.1

# Camera Settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30