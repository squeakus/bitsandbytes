import pygame, sys,os
from pygame.locals import * 

pygame.init() 

window = pygame.display.set_mode((468, 60)) 
pygame.display.set_caption('Monkey Fever') 
screen = pygame.display.get_surface() 

monkey_head_file_name = os.path.join("data","chimp.bmp")

monkey_surface = pygame.image.load(monkey_head_file_name)

screen.blit(monkey_surface, (0,0)) 
pygame.display.flip() 

def input(events): 
   for event in events: 
      if event.type == QUIT: 
         sys.exit(0) 
      else: 
         print event 

while True: 
   input(pygame.event.get())
