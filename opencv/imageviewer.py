import cv
ll = "img00040.jpg"
img = cv.LoadImage(ll)
x1, y1 = 2100, 150
x2, y2 = x1+1920, y1+1080

#cv.NamedWindow("Example", cv.CV_WINDOW_AUTOSIZE )
cv.NamedWindow("Example", cv.CV_WINDOW_NORMAL)
cv.Circle(img, (2100,150), 100, [255,255,255], thickness=3, lineType=3)
cv.Rectangle( img, (x1,y1),(x2, y2),[255,255,255],
              thickness=3, lineType=4);
cv.ShowImage("Example", img )
cv.WaitKey(10000)
cv.DestroyWindow("Example")
