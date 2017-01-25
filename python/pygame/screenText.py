#!/usr/bin/python

import pygame, time
from pygame.locals import *

def show_text(countdown, background, screen):
    for i in range(countdown):
        countText = "Taking picture in: "+ str(countdown-i)
        write_text(countText, background, screen)
        time.sleep(1)
                
    write_text("SNAP!", background, screen)
    time.sleep(2)
    write_text(default_text, background, screen)


def write_text(text, background, screen):
    # set up text variables
    font = pygame.font.SysFont('sawasdee',80)
    back_color = (240,240,240)
    text_color = (10,10,10)

    #code for showing fonts
    #print(pygame.font.get_fonts())
    #font = pygame.font.Font(None, 100)

    #draw background and text
    background.fill(back_color)
    text = font.render(text, True, text_color)
    textpos = text.get_rect()
    textpos.centerx = background.get_rect().centerx
    textpos.centery = background.get_rect().centery
    background.blit(text, textpos)

    # Blit everything to the screen)
    screen.blit(background, (0, 0))
    pygame.display.flip()

def main():
    default_text = "Press any button to take a picture"
    countdown = 5 # second before photo
    pygame.init() # Initialise screen
    screen = pygame.display.set_mode((1280, 600), FULLSCREEN)
    pygame.display.set_caption('photo booth')

    # Fill background
    background = pygame.Surface(screen.get_size())
    background = background.convert()

    write_text(default_text, background, screen)

    #setting up joysticks
    njoy = pygame.joystick.get_count()
    print "Number of joysticks detected = ",njoy
    for j in range(njoy):
        gamepad = pygame.joystick.Joystick(j)
        gamepad.init()


    # Event loop
    while True:
        for event in pygame.event.get():
            if event.type == QUIT: return
            if event.type == KEYDOWN: 
                if event.key == K_ESCAPE:
                    print "esc KEY"
                    return
                if event.key == K_SPACE:
                    show_text(countdown, background, screen)
            #listen for joystick events
            if event.type == JOYBUTTONDOWN:
                show_text(countdown, background, screen)

if __name__ == '__main__': main()
