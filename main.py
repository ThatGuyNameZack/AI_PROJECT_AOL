
import site
import sys 
import cv2

print(site.getsitepackages())  # Print site packages for debugging
 # In case of different version dependencie


cam = cv2.VideoCapture(0)
print("tracking modelue imported.")

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
