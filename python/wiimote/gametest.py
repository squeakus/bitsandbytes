import pygame
import pygame_wiimote # ideally something like this would be part of pygame so the _ would become .
import sys

pygame.init()

# initialze the wiimotes
print 'press 1&2'
pygame_wiimote.init(1, 5) # look for 1, wait 5 seconds
n = pygame_wiimote.get_count() # how many did we get?

if n == 0:
    print 'no wiimotes found'
    sys.exit(1)

wm = pygame_wiimote.Wiimote(0) # access the wiimote object
wm.enable_accels(1) # turn on acceleration reporting

w,h = size = (512,512)
screen = pygame.display.set_mode(size)

run = True

old = [h/2] * 3
maxA = 2.0

colors = [ (255,0,0), (0,255,0), (0,0,255) ]

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print 'quiting'
            run = False
            break
        elif event.type == pygame_wiimote.WIIMOTE_BUTTON_PRESS:
            print event.button, 'pressed'
        elif event.type == pygame_wiimote.WIIMOTE_BUTTON_RELEASE:
            print event.button, 'released'
        elif event.type == pygame_wiimote.WIIMOTE_ACCEL:
            for c in range(3):
                s = int((event.accel[c] * h / maxA + h)/2)
                s = max(0, min(h-1, s))
                pygame.draw.line(screen, colors[c], (w-3, old[c]), (w-2, s))
                old[c] = s
            screen.blit(screen, (-1, 0))
        elif event.type == pygame_wiimote.NUNCHUK_ACCEL:
            print 'n accel', event.accel

    pygame.display.flip()
    pygame.time.wait(10)

wm.disconnect()
