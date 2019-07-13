from math import hypot, pi, cos, sin
from PIL import Image
import numpy as np
import cv2
 
def hough(image, theta_x=360, rho_y=360):
    "Calculate Hough transform."
    print(image.shape)
    height, width = image.shape
    rho_y = int(rho_y/2)*2          #Make sure that this is even
    him = np.zeros((theta_x, rho_y))
 
    rmax = hypot(width, height)
    dr = rmax / (rho_y/2)
    dth = pi / theta_x
    frame = 0

    print("drho", dr, "dtheta", dth)
    for x in range(width):
        for y in range(height):
            col = image[x, y]
            
            new = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.circle(new,(x,y), 5, (0,0,255))
            print(frame, x,y)
            cv2.imwrite(imagename, new)
            frame += 1

            if col == 255: continue
            for tx in range(theta_x):
                th = dth * tx
                r = x*cos(th) + y*sin(th)
                x2 = int(x + (20 * cos(th)))
                y2 = int(y + (20 * sin(th)))
                #print(x,y,x2,y2)
                new = cv2.line(new, (x, y), (x2, y2), (255, 0, 0), 1)
                #imagename = "img{:04}.png".format(frame) 
                #cv2.waitKey(5)
                iry = rho_y/2 + int(r/dr+0.5)
                him[int(iry),int(tx)] += 1

    return him
 
 
def test():
    "Test Hough transform with pentagon."
    im = cv2.imread("pentagon.png", 0)
    him = hough(im)
    cv2.imwrite("houghspace.png", him)
 
 
if __name__ == "__main__": test()
