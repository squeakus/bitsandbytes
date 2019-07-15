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
    modval = 50
    rmax = hypot(width, height)
    dr = rmax / (rho_y/2)
    dth = pi / theta_x
    frame = 0
    once = False

    print("drho", dr, "dtheta", dth)
    for x in range(height):
        for y in range(width):
            col = image[x, y]
            
            # set up a frame for drawing on.
            frame += 1
            imagename = "img{:05}.png".format(frame)
            new = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            if col == 255: 
                if frame % modval == 0:
                    new = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)    
                    cv2.circle(new,(y, x), 3, (0,255,0), -1)
                    new = cv2.resize(new, (360,360))
                    a = np.expand_dims(him, axis = 2)
                    newhough = np.concatenate((a,a,a), axis = 2)
                    vis = np.concatenate((new, newhough), axis=1)
                    cv2.imwrite(imagename, vis)
                continue

            for tx in range(theta_x):
                th = dth * tx
                r = x*cos(th) + y*sin(th)
                x1,y1,x2,y2 = findline(image, y,x, th)

                #
                #cv2.waitKey(5)
                iry = rho_y/2 + int(r/dr+0.5)
                print("tx", tx, "th", th, "r", r, "iry", iry)
                him[int(iry),int(tx)] += 1


                frame += 1
                if (once == False and frame % 3 == 0) or (once == True and frame % 50 == 0):
                    if frame > 50000 and modval < 150:
                        modval += 1
                    once = True
                    imagename = "img{:05}.png".format(frame)
                    new = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    cv2.circle(new,(y, x), 3, (255,0, 0), -1)
                    cv2.line(new, (x1, y1), (x2, y2), (255, 0, 0), 1)

                    a = np.expand_dims(him, axis = 2)
                    newhough = np.concatenate((a,a,a), axis = 2)
                    new = cv2.resize(new, (360,360))
                    cv2.circle(newhough, (int(tx),int(iry)), 3, (0,0,255), -1)
                    vis = np.concatenate((new, newhough), axis=1)

                    cv2.imwrite(imagename, vis)
                    print(imagename)


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
    im = cv2.imread("pentagon.png", 0)
    him = hough(im)
    cv2.imwrite("houghspace.png", him)
 
 
if __name__ == "__main__": test()
