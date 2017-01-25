import os, sys, inspect, thread, time, math
import pygame, Leap
from pygame.locals import QUIT

def main():
    # Create a sample listener and controller
    controller = Leap.Controller()


    pygame.init()
    screen = pygame.display.set_mode((640,480))
    white = (255,255,255)
    black = (0,0,0)
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)

    while True:
        #quit cleanly if excape pressed
        x = 200
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit();

        #draw black screen and get next leap frame
        screen.fill(black);
        frame = controller.frame()
        # Get hands
        for hand in frame.hands:
            if hand.is_left:
                x = hand.palm_position[1]
                pygame.draw.circle(screen, red, (int(x), 200), 20, 1)
            else:
                x = hand.palm_position[1]
                pygame.draw.circle(screen, green, (int(x), 300), 20, 1)
                


        pygame.display.update()

if __name__ == "__main__":
    main()
