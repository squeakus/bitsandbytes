"""A module for plotting results"""

import pylab, pygame, sys
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

filetype = '.png'

def plot_3d(results_list, title):
    """show all results in parallel"""   
    x_range = range(len(results_list[0]))
    fig = plt.figure()
    #plt.title(title)
    axe = Axes3D(fig)
    plt.title(title)
    for idx, result in enumerate(results_list):
        axe.plot(x_range, result, idx)
    plt.show()

def plot_2d(results_list, title):
    """multiple runs single graph"""
    pylab.clf()
    pylab.figure().autofmt_xdate()
    x_range = range(len(results_list[0]))
    for result in results_list:
        pylab.plot(x_range, result)
    pylab.title(title)
    title += filetype    
    pylab.savefig(title)

def boxplot_data(results_list, title):
    pylab.clf()
    pylab.figure(1)
    result_cols = []
    for i in range(len(results_list[0])):
        res = [result[i] for result in results_list]
        result_cols.append(res)
    pylab.boxplot(result_cols)
    pylab.figure(1).autofmt_xdate()
    title += '_boxplot'
    pylab.title(title)
    title += filetype
    pylab.savefig(title)
    
def plot_ave(results_list, title):
    """ show average with error bars"""
    pylab.clf()
    pylab.figure().autofmt_xdate()

    x_range = range(len(results_list[0]))
    err_x, err_y, std_list = [], [], []

    for i in x_range:
        if i % 10 == 0:
            #get average for each generation
            column = []     
            for result in results_list:
                column.append(result[i])
            average = np.average(column)
        
            std_dev = np.std(column)
            err_x.append(i)
            err_y.append(average)
            std_list.append(std_dev)
    pylab.errorbar(err_x, err_y, yerr=std_list)
    title += '_average'
    pylab.title(title)
    title += filetype
    pylab.savefig(title)

def continuous_plot(iterations, grn):
    """Uses pygame to draw concentrations in real time"""
    width, height = size = (600, 600)
    screen = pygame.display.set_mode(size)
    colors = [] # list for protein colors
    conc_list = [] # current concentrations
    extra_list = [] # add variables for user input
    key_list = [] # keyboard inputs
    extra_colors = [(255, 0, 0),
                    (255, 255, 0),
                    (255, 0, 255),
                    (0, 255, 255)]
    key_list.append([pygame.K_UP, pygame.K_DOWN])
    key_list.append((pygame.K_a, pygame.K_z))
    key_list.append((pygame.K_s, pygame.K_x))
    key_list.append((pygame.K_d, pygame.K_c))

    for gene in grn.genes:
        # TF = Blue P = Green EXTRA = Red
        if gene.gene_type == "TF":
            colors.append((0, 0, 255))
        elif gene.gene_type == "P":
            colors.append((0, 255, 0))
        elif gene.gene_type.startswith("EXTRA"):
            extra_list.append({'name':gene.gene_type,
                               'up':False, 'down':False})
            colors.append(extra_colors.pop())
            
        conc_list.append(600-(gene.concentration * 600))

    for _ in range(iterations):
        #check for keypress
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                for idx, key_tuple in enumerate(key_list):
                    if pygame.key.get_pressed()[key_tuple[0]]:
                        extra_list[idx]['up'] = True
                    elif pygame.key.get_pressed()[key_tuple[1]]:
                        extra_list[idx]['down'] = True
                    
            elif event.type == pygame.KEYUP:
                for extra in extra_list:
                    extra['up'] = False
                    extra['down'] = False
                    
        # Update the extra protein concentration 
        for extra in extra_list:
            if extra['up']: 
                grn.change_extra(extra['name'], 0.005)
            if extra['down']: 
                grn.change_extra(extra['name'], -0.005)
                    
        # if extrab_up: 
        #     grn.change_extra("EXTRA_B", 0.005)
        # if extrab_down: 
        #     grn.change_extra("EXTRA_B", -0.005)

        #run grn and get protein concentrations
        results = grn.regulate_matrix(2, False)
        scaled = [int(600-(x * 600)) for x in results]
        old_conc = conc_list
        conc_list = scaled
        
        for idx, conc in enumerate(conc_list):
            pygame.draw.line(screen, colors[idx], 
                             (width-3, old_conc[idx]), 
                             (width-2, conc))

        pygame.display.flip()
        screen.scroll(-1, 0)
        pygame.time.wait(5)
