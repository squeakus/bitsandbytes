from numpy import sin, cos, pi, array
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G =  9.8 # acceleration due to gravity, in m/s^2
L1 = 1.0 # length of pendulum 1 in m
L2 = 1.0 # length of pendulum 2 in m
M1 = 1.0 # mass of pendulum 1 in kg
M2 = 1.0 # mass of pendulum 2 in kg
TIME = 20 # number of seconds it will calculate

def derivs(state, t):

    dydx = np.zeros_like(state)
    dydx[0] = state[1]

    del_ = state[2]-state[0]
    den1 = (M1+M2)*L1 - M2*L1*cos(del_)*cos(del_)
    dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_)
               + M2*G*sin(state[2])*cos(del_) + M2*L2*state[3]*state[3]*sin(del_)
               - (M1+M2)*G*sin(state[0]))/den1

    dydx[2] = state[3]

    den2 = (L2/L1)*den1
    dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_)
               + (M1+M2)*G*sin(state[0])*cos(del_)
               - (M1+M2)*L1*state[1]*state[1]*sin(del_)
               - (M1+M2)*G*sin(state[2]))/den2

    return dydx

# create a time array from 0..100 sampled at 0.1 second steps
dt = 0.05
t = np.arange(0.0, TIME, dt)

# th1 and th2 are the initial angles (degrees)
# w10 and w20 are the initial angular velocities (degrees per second)
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 120.0

rad = pi/180

# initial state
state = np.array([th1, w1, th2, w2])*pi/180.

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)
print len(y)
x1 = L1*sin(y[:,0])
y1 = -L1*cos(y[:,0])

x2 = L2*sin(y[:,2]) + x1
y2 = -L2*cos(y[:,2]) + y1

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
points, = ax.plot([], [], 'ro-',markersize=4)

time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

pointpath = {'x':[],'y':[]}

def init():
    line.set_data([], [])
    points.set_data([], [])
    time_text.set_text('')
    pointpath = {'x':[],'y':[]}

    return line, points, time_text

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    pointpath['x'].append(x2[i])
    pointpath['y'].append(y2[i])

    #    if len(pointpath['x']) > 50:
    #    pointpath['x'].pop(0)
    #    pointpath['y'].pop(0)
        
    
    line.set_data(thisx, thisy)
    points.set_data(pointpath['x'],pointpath['y'])
    time_text.set_text(time_template%(i*dt))
    return line, points, time_text

# ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
#       interval=25, blit=True, init_func=init)

# #ani.save('./double_pendulum.mp4', fps=15, clear_temp=True)
# plt.show()
# fig = plt.figure()
# #plt.xlim(-2, 2)
# #plt.ylim(-2, 2)
# plt.plot(y2)
# plt.ylabel("pendulum height")
# plt.xlabel("time")

# plt.show()

names = ["Angle 1", "Ang. Vel. 1", "Angle 2", "Ang. Vel. 2"]
plt.plot(y[:,0],y[:,1],marker=".")
plt.xlabel(names[0])
plt.ylabel(names[1])
plt.figure()
plt.plot(y[:,0],y[:,2],marker=".")
plt.xlabel(names[0])
plt.ylabel(names[2])
plt.figure()
plt.plot(y[:,0],y[:,3],marker=".")
plt.xlabel(names[0])
plt.ylabel(names[3])
plt.figure()
plt.plot(y[:,1],y[:,2],marker=".")
plt.xlabel(names[1])
plt.ylabel(names[2])
plt.figure()
plt.plot(y[:,1],y[:,3],marker=".")
plt.xlabel(names[1])
plt.ylabel(names[3])
plt.figure()
plt.plot(y[:,2],y[:,3],marker=".")
plt.xlabel(names[2])
plt.ylabel(names[3])
plt.show()
