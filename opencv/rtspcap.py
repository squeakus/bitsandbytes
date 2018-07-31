import cv2 as cv
vcap = cv.VideoCapture("rtsp://192.168.1.199:8554/out.h264")
while(1):
    ret, frame = vcap.read()
    cv.imshow('VIDEO', frame)
    cv.waitKey(1)