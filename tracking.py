import cv2
import config

def TrackingObject():
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam

    if not video_capture.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize the face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(config.FACE_DETECTION_MODEL)

    # Create the tracker
    tracker = cv2.TrackerKCF_create()

    # Read the first frame
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Failed to read frame.")
        return

    # Convert the frame to grayscale (face detection works on grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Manually select the first face as the bounding box
        bbox = tuple(faces[0])  # (x, y, w, h)
        tracker.init(frame, bbox)
        print(f"Tracking face at {bbox}")
    else:
        print("No face detected!")

    # Start tracking loop
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Update the tracker with the new frame
        success, bbox = tracker.update(frame)

        # If tracking is successful, draw the bounding box
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failed", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display the frame with tracking
        cv2.imshow("Tracking", frame)

        # Break the loop when 'p' is pressed
        if cv2.waitKey(1) == ord("p"):
            break

    # Release the camera and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    TrackingObject()
