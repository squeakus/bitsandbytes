from __future__ import print_function
import sys

def main(filename):
    verts = []

    print("Processing", filename)
    infile = open(filename, 'r')
    for line in infile:
        if line.startswith('v '):
             coord = [float(i) for i in line.split(' ')[1:]]
             verts.append(coord)
    minvals = map(min, zip(*verts))
    maxvals = map(max, zip(*verts))
    print("min:", minvals, "max:", maxvals)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: objreader.py <filename>")
        exit()
    main(sys.argv[1])
