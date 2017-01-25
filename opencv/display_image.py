import cv2
import sys
import numpy as np

drawing = False
framecount = 0
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global drawing, framecount
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            print framecount
            framecount += 1
            x2, y2 = x+1920, y+1080
            cv2.rectangle( img, (x,y),(x2, y2),[255,255,255],
                           thickness=3, lineType=4);
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False

if len(sys.argv) != 2: ## Check for error in usage syntax
    print "Usage : python display_image.py <image_file>"
    exit()
else:
    img = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_COLOR) ## Read image file

if img == None: ## Check for invalid input
    print "Could not open or find the image"
else:
    cv2.namedWindow('Display Window',  cv2.WINDOW_NORMAL) ## create window for display
    cv2.imshow('Display Window', img) ## Show image in the window
    print "size of image: ", img.shape ## print size of image
    cv2.setMouseCallback('Display Window',draw_circle)

    while True:
        cv2.imshow('Display Window',img)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows() ## Destroy all windows
