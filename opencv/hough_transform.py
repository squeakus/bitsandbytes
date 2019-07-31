from math import hypot, pi, cos, sin
from PIL import Image
import numpy as np
import cv2
 
def hough(image, theta_x=600, rho_y=600):
    "Calculate Hough transform."
    print(image.shape)
    height, width = image.shape
    rho_y = int(rho_y/2)*2          #Make sure that this is even
    him = np.zeros((theta_x, rho_y))
    rmax = hypot(width, height)
    dr = rmax / (rho_y/2)
    dth = pi / theta_x
    frame = 0
    spincnt = 0
    modval = 50
    fast = False
    print("drho", dr, "dtheta", dth)
    for x in range(height):
        for y in range(width):
            col = image[x, y]
            
            # set up a frame for drawing on.
            frame += 1
            imagename = "img{:07}.png".format(frame)
            new = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            if col == 255: 
                if frame % modval * 3 == 0 and not fast:
                    new = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)    
                    cv2.circle(new,(y, x), 3, (0,255,0), -1)
                    new = cv2.resize(new, (600,600))
                    a = np.expand_dims(him, axis = 2)
                    newhough = np.concatenate((a,a,a), axis = 2)
                    vis = np.concatenate((new, newhough), axis=1)
                    cv2.imwrite(imagename, vis)
                continue

            if fast:
                if frame % (modval/5) == 0:
                    cv2.circle(new,(y, x), 3, (255,0, 0), -1)
                    new = cv2.resize(new, (600,600))
                    a = np.expand_dims(him, axis = 2)
                    newhough = np.concatenate((a,a,a), axis = 2)
                    vis = np.concatenate((new, newhough), axis=1)
                    cv2.imwrite(imagename, vis)

            for tx in range(theta_x):
                th = dth * tx
                r = x*cos(th) + y*sin(th)
                x1,y1,x2,y2 = findline(image, y,x, th)

                #
                #cv2.waitKey(5)
                iry = rho_y/2 + int(r/dr+0.5)
                him[int(iry),int(tx)] += 5

                frame += 1
                if (spincnt == 0 and frame % 3 == 0) or (spincnt > 1 and frame % modval == 0):
                    if spincnt > 20:
                        fast = True
                        continue
                    imagename = "img{:07}.png".format(frame)
                    new = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    cv2.circle(new,(y, x), 3, (255,0, 0), -1)
                    cv2.line(new, (x1, y1), (x2, y2), (255, 0, 0), 1)

                    a = np.expand_dims(him, axis = 2)
                    newhough = np.concatenate((a,a,a), axis = 2)
                    new = cv2.resize(new, (600,600))
                    cv2.circle(newhough, (int(tx),int(iry)), 3, (0,0,255), -1)
                    vis = np.concatenate((new, newhough), axis=1)

                    cv2.imwrite(imagename, vis)
                    print(imagename)

            spincnt += 1

            #exit()

    return him
 

def findline(image, x,y, th):
        x2 = int(x + (300 * cos(th)))
        y2 = int(y + (300 * sin(th)))
        x1 = x + (x - x2)
        y1 = y + (y - y2)
        return x1,y1,x2,y2
 
def test():
    "Test Hough transform with pentagon."
    im = cv2.imread("pentagon2.png", 0)
    him = hough(im)
    cv2.imwrite("houghspace.png", him)
 
 
if __name__ == "__main__": test()
