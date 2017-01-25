"""
This program is used for randomizing/deleting points in a point cloud
"""

import random, sys

def main():
    if len(sys.argv) < 3:
        print "Usage: python cloudparser.py <inputfile> <outputfile>"
        exit()

    infile = sys.argv[1]
    outfile = sys.argv[2]
    points = read_asc(infile)
    #newpoints = randomize_points(points, 5)
    newpoints  = delete_points(points, 0.9)
    write_asc(outfile, newpoints)

def delete_points(points, probability):
    """delete points out of the set with a given probability"""
    newpoints = []

    for point in points:
        x, y, z = point[0], point[1], point[2]
        if random.random() > probability:
            newpoints.append((x, y, z))
    return newpoints

def randomize_points(points, scale):
    """
    Given a set of points, it will move each of the points
    randomly by an amount decided by the scale value
    """
    randpoints = []

    for point in points:
        x, y, z = point[0], point[1], point[2]
        x = x + random.uniform(-scale, scale)
        y = y + random.uniform(-scale, scale)
        z = z + random.uniform(-scale, scale)
        randpoints.append((x, y, z))
    return randpoints

def read_asc(filename):
    """
    This function will read a given ascii point file
    convert the text values to floats and return a list of points
    """
    ascfile = open(filename, 'r')
    points = []

    for line in ascfile:
        if not line.startswith('//'):
            line = line.split(' ')
            x, y, z = float(line[0]), float(line[1]), float(line[2])
            points.append((x, y, z))

    ascfile.close()
    return points

def write_asc(filename, points):
    """
    Writes the points out to a file in ascii format
    """
    outfile = open(filename, 'w')

    for point in points:
        x, y, z = str(point[0]), str(point[1]), str(point[2])
        outfile.write(x + ' ' + y + ' ' + z + '\n')
    outfile.close()

if __name__ == '__main__':
    main()
