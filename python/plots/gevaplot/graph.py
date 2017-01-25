"""A module for plotting results"""

import pylab, sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

FILETYPE = '.png'

def plot_3d(result_list, title, colors=None):
    """show all results in parallel"""   
    x_range = range(len(result_list[0]))
    fig = plt.figure()
    axe = Axes3D(fig)
    plt.title(title)
    for idx, result in enumerate(result_list):
        if colors == None:
            axe.plot(x_range, result, idx)
        else:
            axe.plot(x_range, result, idx, color=colors[idx])
    plt.show()

def plot_2d(result_list, title, colors=None):
    """multiple runs single graph"""
    pylab.clf()
    pylab.figure().autofmt_xdate()
    x_range = range(len(result_list[0]))
    for idx, result in enumerate(result_list):
        if colors == None:
            pylab.plot(x_range, result)
        else:
            pylab.plot(x_range, result, colors[idx])
    pylab.title(title)
    title = 'graphs/' +title + FILETYPE
    print "saving fig", title
    pylab.savefig(title)

def boxplot_data(result_list, title):
    pylab.clf()
    pylab.figure(1)
    result_cols = []
    for i in range(len(result_list[0])):
        res = [result[i] for result in result_list]
        result_cols.append(res)
    pylab.boxplot(result_cols)
    pylab.figure(1).autofmt_xdate()
    title = title + '_boxplot'
    pylab.title(title)
    filename  = 'graphs/' + title + FILETYPE
    pylab.savefig(filename)
    
def plot_ave(result_list, title, multiple=False):
    """ show average with error bars"""
    pylab.clf()
    pylab.figure().autofmt_xdate()

    x_range = range(len(result_list[0]))
    err_x, err_y, std_list = [], [], []

    for i in x_range:
        if i % 10 == 0:
            #get average for each generation
            column = []     
            for result in result_list:
                column.append(result[i])
            average = np.average(column)
        
            std_dev = np.std(column)
            err_x.append(i)
            err_y.append(average)
            std_list.append(std_dev)
    pylab.errorbar(err_x, err_y, yerr=std_list)
    title += '_average'
    pylab.title(title)
    filename  = 'graphs/' + title + FILETYPE
    pylab.savefig(filename)

def multi_plot(results, title, legend=None):
    pylab.clf()
    pylab.figure().autofmt_xdate()
    #line cycler for bw graphs
    # lines = ["-","--","-.",":"]
    # linecycler = cycle(lines)

    x_range = range(len(results[0][0]))
    for result_list in results:
        result_list = np.array(result_list)
        average = np.mean(result_list,axis=0)
        std_dev = np.std(result_list, axis=0)
        pylab.plot(average)

        #do the error bars
        err_x, err_y, std_list = [], [], []
        for i in x_range:
            #errorbars every 5 samples            
            if i % 5 == 0:
                err_x.append(i)
                err_y.append(average[i])
                std_list.append(std_dev[i])
        pylab.errorbar(err_x, err_y, yerr=std_list,fmt=None,ecolor='k')

    if not legend ==  None:
        pylab.legend(legend,'best')
        
    pylab.title(title)
    #pylab.ylim(110,126)
    pylab.ylim(100,126)
    pylab.xlabel("generations")
    pylab.ylabel("fitness")
    
    filename  = title + FILETYPE
    pylab.savefig(filename)
