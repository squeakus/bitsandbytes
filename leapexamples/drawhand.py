import os, sys, inspect, time, math
import pygame, Leap
from pygame.locals import QUIT

def predict(oldx, oldy, newx, newy):
    diffx = int((newx - oldx) * 5)
    diffy = int((newy - oldy) * 5)
    if not diffx == 0:
        print "px",diffx, "py",diffy
    predx = newx + diffx
    predy = newy + diffy
    return predx, predy

def main():
    # Create a sample listener and controller
    controller = Leap.Controller()


    pygame.init()
    screen = pygame.display.set_mode((800,600))
    white = (255,255,255)
    black = (0,0,0)
    red = (255,0,0)
    green = (0,255,0)
    blue = (0,0,255)
    leftx, rightx = 400, 400
    lefty, righty = 300, 300

    while True:
        #quit cleanly if excape pressed
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit();

        #draw black screen and get next leap frame
        screen.fill(black);
        frame = controller.frame()
        oldleftx, oldrightx = leftx, rightx 
        oldlefty, oldrighty = lefty, righty

        # Get hands
        for hand in frame.hands:
            if hand.is_left:
                leftx = 400 + int(hand.palm_position[0])
                lefty = 500 - int(hand.palm_position[1])
            else:
                rightx = 400 + int(hand.palm_position[0])
                righty = 500 - int(hand.palm_position[1])

        predx, predy = predict(oldleftx, oldlefty, leftx, lefty)
        pygame.draw.circle(screen, blue, (predx, predy), 20, 1)
        pygame.draw.circle(screen, red, (leftx, lefty), 20, 1)

        predx, predy = predict(oldrightx, oldrighty, rightx, righty)
        pygame.draw.circle(screen, blue, (predx, predy), 20, 1)
        pygame.draw.circle(screen, green, (rightx, righty), 20, 1)

        pygame.display.update()

if __name__ == "__main__":
    main()
