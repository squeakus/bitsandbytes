# drawCircleArcExample.py     P. Conrad for CS5nm, 10/31/2008
#  How to draw an arc in Pygame that is part of a circle

import pygame
from pygame.locals import QUIT
from sys import exit

import math

pygame.init()
screen = pygame.display.set_mode((640,480))
white = (255,255,255)
black = (0,0,0)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
x = 200
x_up, x_down = False, False


while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit(); exit();
        elif event.type == pygame.KEYDOWN:
            if pygame.key.get_pressed()[pygame.K_UP]:
                x_up = True
                print "UP"
            if pygame.key.get_pressed()[pygame.K_DOWN]:
                x_down = True
                print "DOWN"
        elif event.type == pygame.KEYUP:
            print "KEYUP"
            x_up, x_down = False, False

    if x_up: x+= 1
    if x_down: x -=1


    screen.fill(black);
    pygame.draw.circle(screen, red, (x,200), 20, 1)
    pygame.display.update()
