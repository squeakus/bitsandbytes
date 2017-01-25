import time, gc
import numpy as np
import pandas as pd
#sort out prevx

def main():
    points, coords = pandamaxmin("damecleaned.xyz")
    subdivided = subdivide(points, coords)
    write_new(subdivided)
    

def subdivide(points, coords, fraction=5):
    subdivided = {}
    print "number of points", len(points)
    starttime = time.time()
    prevx, prevy = coords['minx'], coords['miny']
    coords['fracx'] =  (coords['maxx'] - coords['minx']) / fraction
    coords['fracy'] =  (coords['maxy'] - coords['miny']) / fraction

    for index, row in points.iterrows():
        for x in range(1, fraction):
            curx = coords['minx'] + (coords['fracx'] * x)
            if prevx < row['x'] and row['x'] < curx:
                name = "x"+str(x)
                if not name in subdivided:
                    subdivided[name] = []
                subdivided[name].append(row)
            prevx = curx
    timetaken = time.time() - starttime 
    print "subdivide taken:", timetaken
    return subdivided

def write_new(subdivided):
    print subdivided
    starttime = time.time()
    for key in subdivided:
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
    points = pd.read_csv(filename, header=None, delim_whitespace=True)
    points.columns = ['x', 'y','z','xrgb', 'yrgb', 'zrgb', 'trans']
    maxvals = points.max(axis=0)
    minvals = points.min(axis=0)
    coords = {'maxx': maxvals['x'], 'maxy': maxvals['y'], 'maxz':maxvals['z'],
              'minx': minvals['x'], 'miny': minvals['y'], 'minz':minvals['z']}
    return points, coords

if __name__=='__main__':
    main()
