import curses, time
from Queue import Queue

class gridmap():
    def __init__(self):
        width, height = 10, 10
        wall = [(4,0),(4,1),(4,2),(4,3)]
        all_nodes = init_nodes(width, height, wall)
        start =(1,1)
        frontier = Queue()
        frontier.put(start)
        came_from = {}
        came_from[start] = None

def main(screen):
    curses.curs_set(0)
    
    width, height = 10, 10
    wall = [(4,0),(4,1),(4,2),(4,3)]
    all_nodes = init_nodes(width, height, wall)
    start =(1,1)
    frontier = Queue()
    frontier.put(start)
    came_from = {}
    came_from[start] = None

    while not frontier.empty():
        current = frontier.get()
        for nextnode in neighbors(current, all_nodes):
            if nextnode not in came_from:
                frontier.put(nextnode)
                came_from[nextnode] = current
                x,y = nextnode
                drawmap(x, y, screen, frontier, came_from)
    drawpath(came_from, screen)
    curses.endwin()

def init_nodes(width, height, wall):
    all_nodes = []
    for x in range(width):
        for y in range(height):
            if not (x,y) in wall:
                all_nodes.append((x, y))
    return all_nodes

def neighbors(node, all_nodes):
    dirs = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    result = []
    for dir in dirs:
        neighbor = (node[0] + dir[0], node[1] + dir[1])
        if neighbor in all_nodes:
            result.append(neighbor)
    return result

def drawpath(came_from, screen):
    start =(1, 1)
    current = (9,5)
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
        
    screen.border(0)
    for elem in path:
        x,y = elem
        screen.addstr(x+1, y+1, '*') 
    screen.addstr(0,5, "Done")
    screen.refresh()    
    screen.getch()
    
def drawmap(currentx, currenty, screen, frontier, came_from):
    screen.clear()

    screen.border(0)
    counter = 0
    for key in came_from:
        if came_from[key] != None:
            x, y = key
            screen.addstr(x+1, y+1, "v") 
            counter += 1
    #for idx, elem in enumerate(frontier.queue):
    #    x,y = elem
    #    screen.addstr(x+1, y+1, str(idx)) 

        
    screen.addstr(currentx+1, currenty+1, "X")
    screen.addstr(2, 2, "S")

    #visitstr =  "visited:" + str(len(visited))
    screen.addstr(0,5, "length:" + str(len(came_from))+" drawing:"+str(counter))
    screen.refresh()
    time.sleep(0.1)   
    #screen.getch()
    
    
if __name__=='__main__':
    curses.wrapper(main)
