import numpy as np
import cv2

def main(filename):
    # Load an color image in grayscale
    img = cv2.imread(filename, 0)
    show(img)

    # standard binary thresholding
    # ret,thresh = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    # show(thresh)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    show(thresh)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(thresh,kernel, iterations = 1)
    show(dilation)

    # Extract contours and filter by length
    longest = 0
    im2, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # find the longest contour
    for contour in contours:
        if len(contour) > longest:
            longest = len(contour)
    # fill small contours as black
    for contour in contours:    
        if len(contour) < (longest/2):
            cv2.drawContours(dilation, [contour], 0, 0, cv2.FILLED)
    # fill the bigger ones with white
    for contour in contours:
        if len(contour) > (longest/2):

            cv2.drawContours(dilation, [contour], 0, 255, cv2.FILLED)

    # now it has been cleaned up regenerate the contours
    im2, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # find the longest contour
    longest = 0
    for contour in contours:
        if len(contour) > longest:
            longest = len(contour)


    # Use as mask to extract ROI from the topographic image
    topname = filename.replace('FHC', 'TOP')
    top = cv2.imread(topname, 0)
    topmask = cv2.bitwise_and(top,top, mask = dilation)
    show(topmask)

    # create a color image from the mask which we can draw on
    color = cv2.cvtColor(topmask, cv2.COLOR_GRAY2RGB)
    for contour in contours:
        if len(contour) > (longest/2):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour,True)
            print("area: ", area, "perimeter:", perimeter)
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,0),2)
            show(color)
            #unaligned rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(color,[box],0,(0,0,255),2)
            show(color)
            #best fit circle
            (x,y),radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(color,center,radius,(255, 0, 0),2)
            show(color)


def show(image, imagename="out.png"):
    cv2.imshow('image',image)

    k = cv2.waitKey(0)
    if k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite(imagename,img)
        cv2.destroyAllWindows()
    else:
          cv2.destroyAllWindows()
if __name__ == "__main__":
    main('sphereFHC.jpg')
    main('bigFHC.jpg')
    main('medFHC.jpg')
    main('bigFHC.jpg')
    main('lineFHC.jpg')