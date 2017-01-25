#! /usr/bin/python
from openni import *
import pygame, sys
from pygame.locals import QUIT

#global hands, ugly but temporary
hands_generator = HandsGenerator()
position = [0,0,0]

def main():
    #initialise kinect
    context = Context()
    context.init()

    depth_generator = DepthGenerator()
    depth_generator.create(context)
    depth_generator.set_resolution_preset(RES_VGA)
    depth_generator.fps = 30

    gesture_generator = GestureGenerator()
    gesture_generator.create(context)
    gesture_generator.add_gesture('Wave')
    hands_generator.create(context)

    # Register the callbacks
    gesture_generator.register_gesture_cb(gesture_detected, gesture_progress)
    hands_generator.register_hand_cb(create, update, destroy)

    # Start generating
    context.start_generating_all()
    print 'Make a Wave to start tracking...'
    
    #set up a display
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
        # update the gesture data
        context.wait_any_update_all()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); sys.exit();
                
        print position
        leftx = 400 + int(position[0])
        lefty = 500 - int(position[1])


        #draw black screen
        screen.fill(black);
        pygame.draw.circle(screen, red, (leftx, lefty), 20, 1)
        pygame.display.update()

# Declare the callbacks
def gesture_detected(src, gesture, id, end_point):
    print "Detected gesture:", gesture
    hands_generator.start_tracking(end_point)

def gesture_progress(src, gesture, point, progress): pass

def create(src, id, pos, time):
    global position
    position = pos
    print 'Create ', id, pos

def update(src, id, pos, time):
    global position
    position = pos
    print 'Update ', id, pos

def destroy(src, id, time):
    print 'Destroy ', id

def predict(oldx, oldy, newx, newy):
    diffx = int((newx - oldx) * 5)
    diffy = int((newy - oldy) * 5)
    if not diffx == 0:
        print "px",diffx, "py",diffy
    predx = newx + diffx
    predy = newy + diffy
    return predx, predy

if __name__=='__main__':
    main()
