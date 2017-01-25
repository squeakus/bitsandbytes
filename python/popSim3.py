import pygame

def nextPop(pop,rate):
    nextPop = pop*rate*(1-pop)
    return nextPop


initPop = 0.5
growthRate = 3
generations = 50

w,h = size = (600,600)
screen = pygame.display.set_mode(size)
colors = [ (255,0,0)]
pop = nextPop(initPop,growthRate)

for i in range(generations):
    prev_pop = pop
    pop = nextPop(pop,growthRate)
    print "prev", prev_pop, "next",pop
    if i < 200:
        pop = i
        prev_pop = i - 1
        pygame.draw.line(screen, colors[0], (w-3, prev_pop*h), (w-2, pop *h))
    pygame.display.flip()
    #screen.blit(screen, (-1, 0))
    screen.scroll(-1,0)
    pygame.time.wait(10)

