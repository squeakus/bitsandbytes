import sys, pygame, time
from pygame.locals import *

pygame.init()

size = 300, 300

screen = pygame.display.set_mode(size)

# Fill background
background = pygame.Surface(screen.get_size())
background = background.convert()
background.fill((250, 250, 250))

njoy = pygame.joystick.get_count()
print "Number of joysticks detected = ",njoy

for j in range(njoy):
    gamepad = pygame.joystick.Joystick(j)
    gamepad.init()
    print "Joystick #",j+1,"(", gamepad.get_name(),")"
    print " nb of buttons = ", gamepad.get_numbuttons()
    print " nb of mini joysticks = ", gamepad.get_numhats()
    print " nb of trackballs = ", gamepad.get_numballs()
    print " nb of axes = ", gamepad.get_numaxes()

clock = pygame.time.Clock()
keepGoing = True

# Display some text
#font = pygame.font.Font(None, 36)
#text = font.render("Hello There", 1, (10, 10, 10))
#textpos = text.get_rect()
#textpos.centerx = background.get_rect().centerx
#background.blit(text, textpos)

while keepGoing:
    clock.tick(30)

    for event in pygame.event.get():

#	if event.type == JOYAXISMOTION:
#            print "Joystick:", event.joy, " axis:", event.axis, " value:", event.value

#	elif event.type == JOYHATMOTION:
#		print "Joystick:", event.joy, " hat:", event.hat, " value:", event.value

	if event.type == JOYBUTTONDOWN:
            countdown = 10;
            for i in range(countdown):
                print "taking picture in", str(countdown-i)
                time.sleep(1)
            print "Joystick:", event.joy, " button", event.button, "pressed"

	elif event.type == JOYBUTTONUP:
		print "Joystick:", event.joy, " button", event.button, "released"

	elif event.type == pygame.KEYDOWN:
		keyName = pygame.key.name(event.key)
		print "key pressed:", keyName
		if event.key == pygame.K_ESCAPE:
			keepGoing = False

	elif event.type == pygame.MOUSEBUTTONDOWN:
		print "mouse down:", pygame.mouse.get_pos()

	elif event.type == pygame.MOUSEBUTTONUP:
		print "mouse up:", pygame.mouse.get_pos()

	if event.type == pygame.QUIT:
		keepGoing = False

pygame.quit()
