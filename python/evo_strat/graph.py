import pylab
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


def plot_3d(generations, results_list):    
    x = range(generations)
    fig = plt.figure()
    ax = Axes3D(fig)

    for idx, result in enumerate(results_list):
        print "x",len(x),"y",len(result), idx
        ax.plot(x, result, idx)
    plt.show()

def plot_ave(generations, results_list):
    x = range(generations)
    ave_list, std_list = [], []
    err_x, err_y = [], []
    for i in range(len(results_list[0])):
        tmp = []
        
        for result in results_list:
            tmp.append(result[i])

        average = np.average(tmp)

        if i % 10 == 0:
            std_dev = np.std(tmp)
            err_x.append(i)
            err_y.append(average)
            std_list.append(std_dev)

        ave_list.append(average)

    pylab.plot(x,ave_list)
    pylab.errorbar(err_x, err_y, yerr=std_list)
    pylab.show()


