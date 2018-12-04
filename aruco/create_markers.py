import cv2
import cv2.aruco as aruco
import os
 
'''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
'''
 
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
if not os.path.isdir("./images"):
    os.mkdir("./images")
# second parameter is id number
# last parameter is total image size
for i in range(250):

    img = aruco.drawMarker(aruco_dict, i, 50)
    fname = "./images/6x6_{:03d}.jpg".format(i)
    print(fname)
    cv2.imwrite(fname, img)
 