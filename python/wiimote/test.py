import cwiid
print 'place wiimote in discoverable mode (press 1 and 2)...'
wiimote = cwiid.Wiimote()
wiimote.rpt_mode = cwiid.RPT_ACC
#wiimote.state dict now has an acc key with a three-element tuple
#print 'pitch: %d' % (wiimote.state['acc'][cwiid.Y])
print "state:", wiimote.state
