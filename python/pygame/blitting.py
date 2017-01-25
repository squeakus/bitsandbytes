import sys, pygame


w,h = size = (600,600)
x = 300
x_up, x_down = False, False

screen = pygame.display.set_mode(size)
colors = [ (255,0,0)]

for i in range(600):
    prev_x = x
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
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


    pygame.draw.line(screen, colors[0], (w-3, prev_x-1), (w-2, x))
    pygame.display.flip()
    #screen.blit(screen, (-1, 0))
    screen.scroll(-1,0)
    pygame.time.wait(10)

