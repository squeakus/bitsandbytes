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

    # find max and min
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

    # now scale the bastids!
    for filename in siftfiles:
        infile = open(filename, 'r')
        outfilename = filename.rstrip('.sift') + '.bsift'
        outfile = open(outfilename, 'w')

        for line in infile:
            linearray = line.split()
            if len(linearray) == 128:
                newline = ""
                intdescs = map(float, linearray)
                for elem in intdescs:
                    normed = (elem - descmin)/(descmax- descmin)
                    scaled = int(255 * normed)
                    newline += str(scaled) + " "
                outfile.write(newline + '\n')
            else:
                outfile.write(line)
        outfile.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("Usage %s <image_dir>" % sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])
