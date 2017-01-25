import glob
import sys

def main(foldername):
    foldername = foldername.rstrip('/')
    globexpr = "./"+foldername+"/*.sift"
    print "searching", globexpr
    siftfiles = glob.glob(globexpr)
    print "found", siftfiles
    descmin = float('inf')
    descmax = -float('inf')


    for filename in siftfiles:
        infile = open(filename, 'r')
        for line in infile:
            line = line.split()
            if len(line) == 128:
                intdescs = map(int, line)
                for elem in intdescs:
                    if descmax < elem:
                        descmax = elem
                    if descmin > elem:
                        descmin = elem

    print "min", descmin, "max", descmax


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("Usage %s <image_dir>" % sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])
