import main #so it will connect to main.yp
import cv2
import dnn

def TrackingObject():

    video_capture = cv2.VideoCapture()
    tracker = cv2.TrackerKCF_create()

    ret, frame = video_capture.read()

    if not ret:
        print("fail message")
        return 

    #box bounding secara manual not yet auto
    bbox = cv2.selectROI("FRAME", frame, fromCenter=False, showCrosshair=True)
    tracker.init(frame, bbox)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

    #update the tracker
        success, bbox = tracker.update(frame)

        if success:
            p1 = (int(bbox[0], int(bbox[1])))
            p2 = (int(bbox[0] + int(bbox[2], int(bbox[1]+bbox[3]))))
            cv2.rectangleframe(frame, p1, p2, (255,0,0),2,1)
        
        cv2.imshow("Tracking", frame)

video_capture.release()
cv2.destroyAllWindows()

        
        
