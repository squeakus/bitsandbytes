#!/opt/local/bin/python2.4

import pygame, time
import subprocess
from pygame.locals import *

default_text = "Press any button to take a picture"
img_counter = 0

def execute_command(cmd):
     process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
				stdin=subprocess.PIPE)
	
def show_text(countdown, background, screen):
    global img_counter
    for i in range(countdown):
        countText = "Taking picture in: "+ str(countdown-i)
        write_text(countText, background, screen)
        #if i == 2:
    	#	execute_command("gphoto2 --capture-image-and-download --force-overwrite")    
        time.sleep(1)          
    write_text("CHEESE!", background, screen)
    time.sleep(2)
    write_text(default_text, background, screen)
    format_counter = "%04d" % img_counter
    jpg_name = "./photos/img_"+str(format_counter)+".jpg"
    cr2_name = "./photos/img_"+str(format_counter)+".cr2"
    print "saving image:" + str(jpg_name) 
    #execute_command("mv capt0000.cr2 "+ cr2_name, True)
    #execute_command("mv capt0000.jpg "+ jpg_name, True)	
    img_counter += 1

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
    global img_counter
    count_file = open('counter.txt', 'r')
    count_line = count_file.readline()
    img_counter = int(count_line)
    print "starting at image:", img_counter
    count_file.close()

    countdown = 5 # second before photo
    pygame.init() # Initialise screen
    screen = pygame.display.set_mode((1280, 600), FULLSCREEN)
    pygame.display.set_caption('photo booth')

    # Fill background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    execute_command("killall PTPCamera")
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
