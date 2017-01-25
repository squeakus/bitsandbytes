from graph import *
import curses
from Queue import Queue

width, height = 20, 5
gridmap = [[0 for x in xrange(width)] for x in xrange(height)]
screen = curses.initscr()
start =(1,1)
frontier = Queue()
frontier.put(start)
visited = {}
visited[start] = True

def drawmap(currentx, currenty):
    screen.border(0)
    for y, row in enumerate(gridmap):
        for x, val in enumerate(row):
            if val == 0:
                screen.addstr(y+1, x+1, "0")
            elif val == 1:
                screen.addstr(y+1, x+1, "V")
    screen.addstr(currentx+1, currenty+1, "*")
    visitstr =  "visited:" + str(len(visited))
    screen.addstr(20,5, visitstr)

    screen.refresh()    
    screen.getch()
    
while not frontier.empty():
    
    current = frontier.get()
    for nextnode in neighbors(current):
        if nextnode not in visited:
            frontier.put(nextnode)
            visited[nextnode] = True
            x,y = nextnode
            gridmap[x][y] = 1
            drawmap(x,y)


curses.endwin()
