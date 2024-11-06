import cv2
from matplotlib import pyplot as plt

cam = cv2.videoCapture(0)

ret, frame = cam.read()

