import matplotlib, time
matplotlib.use('GTKAgg') # do this before importing pylab
import matplotlib.pyplot as plt 

initPop = 0.5
growthRate = 3.8
generations = 500
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('generation')
ax.set_ylabel('population size')
plt.title("Rate "+str(growthRate))
line, = ax.plot([], [], animated=True, lw=2)
ax.set_ylim(0, 1.1)
ax.set_xlim(0, 100)
ax.grid()


def nextPop(pop,rate):
    nextPop = pop*rate*(1-pop)
    return nextPop


def animate():
    xdata, ydata = [], []
    background = fig.canvas.copy_from_bbox(ax.bbox)
    pop = nextPop(initPop,growthRate)
    xdata.append(0)
    ydata.append(pop)
    for i in range(generations):
        xmin, xmax = ax.get_xlim()
        pop = nextPop(pop,growthRate)
        if i >= xmax - 10:
            xdata.pop(0)
            ydata.pop(0)
            ax.set_xlim(1+xmin,1+xmax)
            fig.canvas.draw()
            background = fig.canvas.copy_from_bbox(ax.bbox)
        xdata.append(i)
        ydata.append(pop)

        line.set_data(xdata, ydata)  # update the data
        ax.draw_artist(line)
        fig.canvas.blit(ax.bbox)

import gobject
print "adding idle"
gobject.idle_add(animate)
print "showing"
plt.show()

