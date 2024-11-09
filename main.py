import tracking
import config

import site
site.getsitepackages()
import sys #incase of different version


import cv2
from matplotlib import pyplot as plt


#img cv2.imread("") read path files depends on th epath file you use
#were lacking the video camera settings
#path for the kaggle dataset
#we could try to use server method for data set


cam = cv2.VideoCapture(0)


# if ret:
#     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.show()

while True:
    ret, frame = cam.read()
    
    
    cv2.imshow('camera', frame)
    
    if cv2.waitKey(1)==ord("p"):
        break #it closes the camer if i i press p.

#plt.show()
cam.release()
cv2.destroyAllWindows()

    