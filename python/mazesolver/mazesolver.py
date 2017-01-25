#!/usr/bin/env python

import sys

from Queue import Queue
from PIL import Image

start = (400,984)
end = (398,25)

def iswhite(value):
    if value[0] >200:
        if value[1] > 200:
            if value[2] > 200:
                return True

def getadjacent(n):
    x,y = n
    return [(x-1,y),(x,y-1),(x+1,y),(x,y+1)]

def BFS(start, end, pixels, base):
    pathcount = 1
    imgcount = 0
    queue = Queue()
    queue.put([start]) # Wrapping the start tuple in a list

    while not queue.empty():

        path = queue.get() 
        pixel = path[-1]

        if pixel == end:
            return path

        for adjacent in getadjacent(pixel):
            x,y = adjacent
            if iswhite(pixels[x,y]):
                pixels[x,y] = (127,127,127) # see note
                new_path = list(path)
                new_path.append(adjacent)
                queue.put(new_path)
                pathcount += 1
                if pathcount % 100 == 0:
                    base.save('img%04d.jpg' % imgcount)
                    imgcount += 1


    print "Queue has been exhausted. No answer was found."


if __name__ == '__main__':

    # invoke: python mazesolver.py <mazefile> <outputfile>[.jpg|.png|etc.]
    base_img = Image.open(sys.argv[1])
    base_img = base_img.convert('RGB')
    base_pixels = base_img.load()

    path = BFS(start, end, base_pixels, base_img)

    path_img = Image.open(sys.argv[1])
    path_pixels = path_img.load()

    for position in path:
        x,y = position
        path_pixels[x,y] = (255,0,0) # red

    path_img.save(sys.argv[2])
