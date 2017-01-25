#!/usr/bin/env python

import nxt.locator, time
from nxt.motor import *

def rotateby(b, degrees):
    m_b = Motor(b, PORT_B)
    m_c = Motor(b, PORT_C)
    m_b.turn(100, degrees)    
    #m_c.turn(5, degrees)
    #m_c.turn(-100, degrees)

print "waiting for brick"
b = nxt.locator.find_one_brick()
print "found"

#2160 degrees in  a full circle
for i in range(10):
    rotateby(b, 360)

