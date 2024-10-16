# import necessary packages
import imutils
import cv2
import pantilthat as pth

class ObjCenter:
    def __init__(self, haarPath):
        # load OpenCV's Haar cascade face detector
        self.detector = cv2.CascadeClassifier(haarPath)
        self.color = (0,0, 255)
        self.testcoord = [160, 120]
        self.flip = 0

    def update(self, frame, frameCenter):
        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # self.flip += 1
        # if self.flip < 100:
        #   return ((320,240), None)
        # elif self.flip < 200:
        #   return (((160,120), None))
        # elif self.flip < 300:
        #   self.flip = 0
        #   return (((160,120), None))
        # if self.flip:
        #   self.testcoord[0] -= 1
        # else:
        #   self.testcoord[0] += 1
        # if self.testcoord[0] > 170:
        #   self.flip = True
        # if self.testcoord[0] < 150:
        #   self.flip = False
        # print(self.testcoord)
        # return (self.testcoord, (self.testcoord[0], self.testcoord[1], 10, 10))

        # detect all faces in the input frame
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.05,
            minNeighbors=9, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
 

        # check to see if a face was found
        if len(rects) > 0:
            # extract the bounding box coordinates of the face and
            # use the coordinates to determine the center of the
            # face
            (x, y, w, h) = rects[0]
            faceX = int(x + (w / 2))
            faceY = int(y + (h / 2))

            # color the error
            pth.set_all(255,0,0)
            if (faceX - frameCenter[0]) > 10:
                pth.set_pixel(0, 255, 255, 255)
            if (faceX - frameCenter[0]) > 30:
                pth.set_pixel(1, 255, 255, 255)
            if (faceX - frameCenter[0]) > 50:
                pth.set_pixel(2, 255, 255, 255)
            if (faceX - frameCenter[0]) < -10:
                pth.set_pixel(7, 255, 255, 255)
            if (faceX - frameCenter[0]) < -30:
                pth.set_pixel(6, 255, 255, 255)
            if (faceX - frameCenter[0]) < -50:
                pth.set_pixel(5, 255, 255, 255)

            pth.show()


            # print("face detected centroid", faceX, faceY)
            # return the center (x, y)-coordinates of the face
            return ((faceX, faceY), rects[0])
 
        # otherwise no faces were found, so return the center of the
        # frame
        pth.clear()
        pth.show()
        return (frameCenter, None)