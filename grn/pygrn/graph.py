"""A module for plotting results"""

import pylab, pygame, sys
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def plot_3d(results_list):
    """show all results in parallel"""   
    x_range = range(len(results_list[0]))
    fig = plt.figure()
    axe = Axes3D(fig)

    for idx, result in enumerate(results_list):
        axe.plot(x_range, result, idx)
    plt.show()

def plot_2d(results_list, idx):
    pylab.clf()
    """multiple runs single graph"""
    fig_name = 'concentrations/seed%03d.png' % idx
    x_range = range(len(results_list[0]))
    for result in results_list:
        pylab.plot(x_range, result)
    #pylab.show()
    pylab.savefig(fig_name)

def plot_ave(results_list):
    """ show average with error bars"""
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
    pylab.show()

def continuous_plot(iterations, grn):
    """Uses pygame to draw concentrations in real time"""
    width, height = size = (600,600)
    screen = pygame.display.set_mode(size)
    # order the colors for the TF andP proteins
    colors = []
    conc_list = []
    extra_up, extra_down = False, False

    for gene in grn.genes:
        
        if gene.gene_type == "TF":
            colors.append((0, 0, 255))
        elif gene.gene_type == "P":
            colors.append((0, 255, 0))
        elif gene.gene_type == "EXTRA":
            colors.append((255,0,0))
            prev_extra = 600-(gene.concentration * 600)

        conc_list.append(600-(gene.concentration * 600))

    # add variables for user input

    for i in range(iterations):
        #check for keypress
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if pygame.key.get_pressed()[pygame.K_UP]:
                    extra_up = True
                if pygame.key.get_pressed()[pygame.K_DOWN]:
                    extra_down = True
            elif event.type == pygame.KEYUP:
                extra_up, extra_down = False, False
        if extra_up: 
            grn.change_extra(0.01)
        if extra_down: 
            grn.change_extra(-0.01)
        #run grn and get protein concentration
        results = grn.regulate_matrix(2, False)
        scaled = [int(600-(x * 600)) for x in results]
        old_conc = conc_list
        conc_list = scaled
        
        for idx, conc in enumerate(conc_list):
            pygame.draw.line(screen, colors[idx], 
                             (width-3, old_conc[idx]), 
                             (width-2, conc))

        # if draw_extra:
        #     pygame.draw.line(screen, colors[-1], 
        #                      (width-3, 600-prev_extra-1), 
        #                      (width-2, 600-extra))

        pygame.display.flip()
        #screen.blit(screen, (-1, 0))
        screen.scroll(-1,0)
        pygame.time.wait(5)
