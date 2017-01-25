import pygame
import time
import math
import sys
#import profile


class particle:
    def __init__(self, c_p, p_p, f, r, lp, st, pixelPos, neighbours):
        self.currentPos = c_p #current position vector
        self.prevPos = p_p #previous position vector
        self.forces = f #acceleration and gravity vector
        self.restLength = r
        self.listPosition = lp
        self.stuck = st
        self.pixelPosition = pixelPos
        self.neighbors = neighbours


global t
global pressed
global scaling
global gravity
gravity = 0.005
scaling = 15
pressed = False
t = 0.5 

pygame.init()
window = pygame.display.set_mode((1000, 1000))
pygame.display.set_caption("Cloth simulation - dombou")
white = (255, 255, 255, 255)



def drawGrid(grid):
    for row in grid:
        for particle in row:
            #pygame.draw.circle(window, white, (particle.currentPos[0]*scaling+50, particle.currentPos[1]*scaling+50), 2, 0)
            particle.pixelPosition = (particle.currentPos[0]*scaling+300, particle.currentPos[1]*scaling+300)
            
            neighbors = particle.neighbors
            for neighbor in neighbors:
                point = grid[neighbor[0]][neighbor[1]]
                pygame.draw.aaline(window, white, (particle.pixelPosition[0], particle.pixelPosition[1]), (point.currentPos[0]*scaling+300, point.currentPos[1]*scaling+300), 1)
                
            prevParticle = particle
    grid[9][0].stuck = True 
    grid[0][0].stuck = True
    pygame.display.flip()    
    window.fill((0, 0, 0, 0), None, 0)
        
    
    
def createGrid():
    rows = 10
    columns = 10
    
    global grid
    grid = []
    for x in range(rows):
        grid.append([])
        for y in range(columns):
            currentPos = (x, y)
            grid[x].append(particle(currentPos, currentPos, (0, gravity), 1, (x,y), False, (0,0), []))   

    for row in grid:
        for point in row:
            neighbors = findNeighbors(point.listPosition, grid)
            point.neighbors = neighbors
            

def verlet(grid):
    for row in grid:
        for particle in row:
            if particle.stuck == True:
                particle.currentPos = particle.prevPos
            else:

                c_p = particle.currentPos
                temp = c_p
                p_p = particle.prevPos
                f = particle.forces
                
                fmultbytime = (f[0]*t*t, f[1]*t*t)
                tempminusp_p = (c_p[0]-p_p[0], c_p[1]-p_p[1]) 
                together = (fmultbytime[0]+tempminusp_p[0], fmultbytime[1] + tempminusp_p[1])
                c_p = (c_p[0]+together[0], c_p[1]+together[1])
                particle.currentPos = c_p
                particle.prevPos = temp

  
def GetInput():
    helddown = False
    mousePressed = pygame.mouse.get_pressed()
    for event in pygame.event.get():
        if event.type == pygame.QUIT: quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE: quit()
              
        if mousePressed[0]:        
            if detectMouseCollision():
               particle = detectMouseCollision()
               helddown = True
               
        while helddown == True:
            together()
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    helddown = False
                else:
                    mousepos = pygame.mouse.get_pos()
                    pos = (float((mousepos[0]-300)/scaling), float((mousepos[1]-300)/scaling))

                    particle.currentPos = pos
                    particle.prevPos = pos
                    #particle.stuck = True

  
def detectMouseCollision():
    global found
    global foundParticle
    found = False
    foundParticle = False
    mousePress = pygame.mouse.get_pressed()
    for row in grid:
        for particle in row:
            xPixel = int(round(particle.pixelPosition[0],0))
            yPixel = int(round(particle.pixelPosition[1],0))
            
            surrounding = []
            surrounding.append((xPixel-1, yPixel))
            surrounding.append((xPixel, yPixel-1))
            surrounding.append((xPixel+1, yPixel-1))
            surrounding.append((xPixel, yPixel+1))
            surrounding.append((xPixel+1, yPixel-1))
            surrounding.append((xPixel-1, yPixel-1))
            surrounding.append((xPixel-1, yPixel+1))
            surrounding.append((xPixel+1, yPixel+1))
            
            
            if pygame.mouse.get_pos() in surrounding:
                found = particle
                break
                break
    
    if found:
        foundParticle = grid[found.listPosition[0]][found.listPosition[1]]
        return foundParticle

  
  
def findNeighbors(pointPosition, grid):
    column_limit = len(grid[0])
    row_limit = len(grid)
    possNeighbors = []
    possNeighbors.append([pointPosition[0]-1, pointPosition[1]])
    possNeighbors.append([pointPosition[0], pointPosition[1]-1])
    possNeighbors.append([pointPosition[0]+1, pointPosition[1]])
    possNeighbors.append([pointPosition[0], pointPosition[1]+1])

    
    neigh = []
    for coord in possNeighbors:
        if (coord[0] < 0) | (coord[0] > row_limit-1):
            pass
        elif (coord[1] < 0) | (coord[1] > column_limit-1):
            pass
        else:
            neigh.append(coord)
            
    finalNeighbors = []
    for point in neigh:
        finalNeighbors.append((point[0], point[1]))
    return finalNeighbors
    
def satisfyConstraints():
    for row in grid:
        for point in row:
            if point.stuck == True:
                point.currentPos = point.prevPos
            else:
                neighbors = point.neighbors

                for constraint in neighbors:
                
                    c2 = grid[constraint[0]][constraint[1]].currentPos
                    c1 = point.currentPos
                    delta = (c2[0]-c1[0], c2[1]-c1[1])
                    deltalength = math.sqrt(((c2[0]-c1[0])**2) + ((c2[1]-c1[1])**2))
                    diff = (deltalength - 1.0)/ deltalength
                    #deltasquared = (delta[0]*delta[0], delta[1]*delta[1])
                    #delta = (delta[0]*(1/(deltasquared[0]+1)-0.5), delta[1]*(1/(deltasquared[1]+1)-0.5))
                    
                    #delta*=(restlength*restlength)/(delta*delta+restlength*restlength)-0.5;
                    dtemp = (delta[0]*0.5*diff, delta[1]*0.5*diff)
                    
                    #c1 = (c1[0] + delta[0], c1[1] + delta[1])
                    #c2 = (c2[0] - delta[0], c2[1] - delta[1])
                    c1 = (c1[0] + dtemp[0], c1[1] + dtemp[1])
                    c2 = (c2[0] - dtemp[0], c2[1] - dtemp[1])
                    
                    point.currentPos = c1
                    grid[constraint[0]][constraint[1]].currentPos = c2
                    
          
def together():
    #verlet(grid)
    satisfyConstraints()
    satisfyConstraints()
    satisfyConstraints()


    drawGrid(grid)
          
def run():

    createGrid()
    time.sleep(0.5)
    while True:
        GetInput()
        together()

#profile.run('run()')
run()

