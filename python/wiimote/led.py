import time, cwiid
print 'place wiimote in discoverable mode (press 1 and 2)...'
wm = cwiid.Wiimote()

for i in range(16):
    wm.led = i
    time.sleep(.5)
print dir(cwiid)
