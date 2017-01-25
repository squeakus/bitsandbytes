from math import sin, pi
from time import sleep
from turtle import *
from numpy import arange
import pylab

GA = 9.80665 # Gravitational Acceleration (meters per second squared)
FORM = 'Time={:6.3f}, Angle={:6.3f}, Speed={:6.3f}'

def main():
    length = 9.0            # Of pendulum (meters)
    ngol = - GA / length    # Negative G over L
    total_time = 0.0        # Seconds
    angle = 1.0             # Initial angle of pendulum (radians)
    speed = 2.0             # Initial angular velocity (radians/second)
    time_step = 0.01        # Seconds
    ang_list, speed_list = [], []

    for i in arange(-2.5,2.5,0.2):
        angle = 1.0             # Initial angle of pendulum (radians)
        total_time = 0

        speed = i
        print "plotting speed", i
        while total_time < 15.0:
            total_time += time_step
            speed += ngol * sin(angle) * time_step
            angle += speed * time_step
            if abs(angle) > 2.0 * pi:  angle = 0

            speed_list.append(speed)
            ang_list.append(angle)
            #print(FORM.format(total_time, angle, speed))
            if draw(angle, length): break
        #sleep(time_step)
    pylab.plot(ang_list, speed_list)
    pylab.xlabel("angle")
    pylab.ylabel("speed")

    #pylab.show()
    pylab.savefig("phaseportrait")
        
def init():
    setup()
    mode('logo')
    radians()
    speed(0)
    hideturtle()
    tracer(False)
    penup()

def draw(angle, length):
    if speed() != 0: return True
    clear()
    setheading(angle + pi)
    pensize(max(round(length), 1))
    pendown()
    forward(length * 25)
    penup()
    dot(length * 10)
    home()
    update()

if __name__ == '__main__':
    init()
    main()
    bye()
