import time, gc
import numpy as np
import pandas as pd
#sort out prevx

def main():
    points, coords = maxmin("damecleaned.xyz")
    subdivided = subdivide(points, coords)
    write_new(subdivided)
#pandamaxmin("damecleaned.xyz")

def maxmin(filename):
    pointcount = 0
    starttime = time.time()
    pointfile = open(filename, 'r')
    
    points = []
    coords = {'maxx': float('-inf'), 'maxy': float('-inf'), 'maxz':float('-inf'),
              'minx': float('inf'), 'miny': float('inf'), 'minz':float('inf')}

    gc.disable()    
    for line in pointfile:
        pointcount += 1
        line = line.split()
        x, y, z = float(line[0]), float(line[1]), float(line[2])
        points.append([x,y,z])

        if x > coords['maxx']: coords['maxx'] = x
        elif x < coords['minx']: coords['minx'] = x

        if y > coords['maxy']: coords['maxy'] = y
        elif y <coords['miny']:coords['miny'] = y

        if z > coords['maxz']: coords['maxz'] = z
        elif z < coords['minz']: coords['minz'] = z

    gc.enable()
    print "number of points:", format(pointcount, ",d")
    timetaken = int(time.time() - starttime)
    print "time taken:", timetaken
    return points, coords

def subdivide(points, coords, fraction=5):
    subdivided = {}
    print "pointlength", len(points)
    starttime = time.time()
    prevx, prevy = coords['minx'], coords['miny']
    coords['fracx'] =  (coords['maxx'] - coords['minx']) / fraction
    coords['fracy'] =  (coords['maxy'] - coords['miny']) / fraction
    print "fracx", coords['fracx'], "fracy", coords['fracy']

    for p in points:        
        for x in range(1, fraction):
            curx = coords['minx'] + (coords['fracx'] * x)
            if prevx < p[0] and p[0] < curx:
                print "found!",x
                for y in range(1, fraction):
                    cury = coords['miny'] + (coords['fracy'] * y)
                    if prevy < p[1] and p[1] < cury:
                        name = "x",str(x),"y",str(y)
                        print "found point in", name
                        if not name in subdivided:
                            subdivided[name] = []
                        subdivided[name].append(p)
                    prevy = cury
            prevx = curx
    timetaken = time.time() - starttime 
    print "subdivide taken:", timetaken
    return subdivided

def write_new(subdivided):
    starttime = time.time()
    print "sublen",len(subdivided)
    for key in subdivided:
        print key
        filename = "dame" + key + ".xyz" 
        outfile = open(filename, 'w')
        for p in subdivided[key]:
            x,y,z = str(p[0]), str(p[1]), str(p[2])
            outfile.write(x + ' ' + y + ' ' + z + '\n')
        outfile.close()
    timetaken = time.time() - starttime 
    print "write taken:", timetaken


    
def pandamaxmin(filename):
    starttime = time.time()
    points = pd.read_csv(filename, delim_whitespace=True)
    timetaken = int(time.time() - starttime) 
    print "time taken:", timetaken

    #points.columns = ['x', 'y','z','xrgb', 'yrgb', 'zrgb', 'trans']
    print "points", points
    print "max size:", points.max(1)
    timetaken = int(time.time() - starttime) 
    print "time taken:", timetaken


if __name__=='__main__':
    main()
