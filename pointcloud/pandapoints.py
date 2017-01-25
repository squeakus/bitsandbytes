"""
This program compares pandas and numpy for reading in the point clouds
PANDAS WIN!
"""

import time, gc
import numpy as np
import pandas as pd
#sort out prevx

def main():
    #pandamaxmin("damecleaned.xyz")
    numpymaxmin("damecleaned.xyz")

def numpymaxmin(filename):
    starttime = time.time()
    points = np.loadtxt(filename)
    timetaken = int(time.time() - starttime)
    print "time taken:", timetaken

    # maxvals = points.max(axis=0)
    # minvals = points.min(axis=0)
    # difference =  maxvals - minvals
    # scaled_diff = difference / 5
    # prevvals = minvals
    # currentvals = minvals + scaled_diff
    print "sdiff\n", scaled_diff
    print "\nPrev\n", prevvals
    print "\nCurr\n", currentvals
    
    timetaken = int(time.time() - starttime)
    print "time taken:", timetaken

    
def pandamaxmin(filename):
    starttime = time.time()
    paramnames = ['x', 'y','z','xrgb', 'yrgb', 'zrgb', 'trans']
    points = pd.read_csv(filename, names=paramnames, delim_whitespace=True)
    maxvals = points.max(axis=0)
    minvals = points.min(axis=0)
    difference =  maxvals - minvals
    scaled_diff = difference / 5
    prevvals = minvals
    currentvals = minvals + scaled_diff
    print "sdiff\n", scaled_diff
    print "\nPrev\n", prevvals
    print "\nCurr\n", currentvals
    newpoints = points[(points['y'] < currentvals['y'])]
    # newpoints = points[(points['x'] < currentvals['x']) &
    #                      (points['y'] < currentvals['y'])]
    print "NEW\n",newpoints
    newpoints.to_csv('newpoints.xyz', sep=' ', header=False, index=False)
    
    timetaken = int(time.time() - starttime)
    print "time taken:", timetaken

if __name__=='__main__':
    main()
