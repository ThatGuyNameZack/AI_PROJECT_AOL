
import site
import sys
import cv2

print(site.getsitepackages())  # Print site packages for debugging

cam = cv2.VideoCapture(0)
print("Tracking module imported.")

if not cam.isOpened():
    print("Error: Could not open camera.")
    sys.exit()


#path set for haas tracking
base_dir  = os.path.dirname(os.path.realpath(__file__))
face_cascade_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Main loop to read frames from the camera
while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw white box around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Display the frame in a window named 'camera'
    cv2.imshow('camera', frame)

    # Press 'p' to exit the loop and close the camera
    if cv2.waitKey(1) == ord("p"):
        break

# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
