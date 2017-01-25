from SimpleCV import *

k = Kinect()
while True:
    try:
        img = k.getImage()
        depth = Image(k.getDepthMatrix(), cv2image=True)
        img.sideBySide(depth).show()
        time.sleep(0.2)
    except KeyboardInterrupt:
        break
