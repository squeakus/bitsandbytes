from math import hypot, pi, cos, sin
from PIL import Image
 
 
def hough(im, theta_x=360, rho_y=360):
    "Calculate Hough transform."
    image = im.load()
    width, height = im.size
    rho_y = int(rho_y/2)*2          #Make sure that this is even
    him = Image.new("L", (theta_x, rho_y), 255)
    phim = him.load()
 
    rmax = hypot(width, height)
    dr = rmax / (rho_y/2)
    dth = pi / theta_x
    print("drho", dr, "dtheta", dth)
    for x in range(width):
        for y in range(height):
            col = image[x, y]
            if col == 255: continue
            for tx in range(theta_x):
                th = dth * tx
                r = x*cos(th) + y*sin(th)
                iry = rho_y/2 + int(r/dr+0.5)
                phim[tx, iry] -= 1
        him.save("img"+str(x)+".png")
    return him
 
 
def test():
    "Test Hough transform with pentagon."
    im = Image.open("pentagon.png").convert("L")
    him = hough(im)
    him.save("ho5.bmp")
 
 
if __name__ == "__main__": test()
