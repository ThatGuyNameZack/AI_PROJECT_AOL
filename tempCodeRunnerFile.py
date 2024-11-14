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

face_cascade_path = '/Users/rafisatria/Documents/GitHub/AI_PROJECT_AOL'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Main loop to read frames from the camera
while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #DETECT FACES
        faces = face_cascade.detectMultiScale(gray, 1.1,4)
        
        #whitebox
        
        for(x,y,w,h) in faces:
            sv2.rectangle(frame, (x,y), (x + w, y + h), (255,255,255), 2)
            

    # Display the frame in a window named 'camera'
    cv2.imshow('camera', frame)
    
    # Press 'p' to exit the loop and close the camera
    if cv2.waitKey(1) == ord("p"):
        break
    
    
# Release the camera and close all windows
cam.release()
cv2.destroyAllWindows()
