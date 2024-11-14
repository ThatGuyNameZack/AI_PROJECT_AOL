
import matplotlib

import site
print(site.getsitepackages())  # Print site packages for debugging
import sys  # In case of different version dependencies

import cv2
from matplotlib import pyplot as plt

# Conditional import for tracking
if 'tracking' not in locals():
    import tracking

# Initialize webcam (camera index 0)
cam = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

# Main loop to read frames from the camera
while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the frame in a window named 'camera'
    cv2.imshow('camera', frame)
    
    # Press 'p' to exit the loop and close the camera
    if cv2.waitKey(1) == ord("p"):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
