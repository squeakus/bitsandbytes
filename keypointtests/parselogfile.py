import sys
import numpy as np
import matplotlib.pyplot as plt

def main(filename):
    '''First extract sift information, then create match array'''

    logfile = open(filename,'r')
    siftpoints = []
    for line in logfile:
        if line.startswith('NOTE:'):
            find_matches(logfile, len(siftpoints))
            break

        # parse out sift keypoints
        if line.startswith('#'):
            line = line.split(':')
            siftpoints.append(int(line[1]))

    for idx, siftcnt in enumerate(siftpoints):
        print idx, siftcnt

def find_matches(logfile, imgcnt):
    ''' When the matching function is called, get relevent results'''
    matcharray = [[0 for x in range(imgcnt)] for x in range(imgcnt)]
    percentarray = [[0 for x in range(imgcnt)] for x in range(imgcnt)]
    for line in logfile:
        if 'Image Match finished' in line:
            break

        if 'and' in line:
            if not 'matches' in line:
                print line
                idx1, idx2, inlier, total = parse_match(line)
                matcharray[idx1][idx2] = (inlier, total)
                percentage = int((float(inlier)/float(total))*100)
                percentarray[idx1][idx2] = percentage
    
    indexes = range(imgcnt)
    x, y = np.meshgrid(indexes, indexes)
    intensity = np.array(percentarray)
    print percentarray
    plt.pcolormesh(x, y, intensity)
    plt.colorbar() #need a colorbar to show the intensity scale
    plt.show()

def parse_match(line):
    '''strip indexes and matches out of the string'''
    line = line.split(':')
    idxlist = line[0].split('and')
    matches = line[1].split(',')[0]
    matches = matches.strip()
    matches = matches.lstrip('[')
    matches = matches.rstrip(']')

    matchlist = matches.split('/')
    matchlist = map(int,matchlist)

    for cnt, index in enumerate(idxlist):
        idxlist[cnt] = int(index)
    print "indexes", idxlist
    print "matches", matchlist
    return idxlist[0],idxlist[1], matchlist[0], matchlist[1]
    
if __name__=='__main__':
    if len(sys.argv) < 2:
        print "usage: python %s <filename>" % sys.argv[0]
        exit()
    main(sys.argv[1])
