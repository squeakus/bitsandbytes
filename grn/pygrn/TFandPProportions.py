#!/usr/bin/env python
import sys # for argv
import random # for random()

from BanzhafModel import *

def main():
    """
    Proportions of P or TF genes
    """
    # Manage arguments
    if(len(sys.argv) != 2):
        print "Usage: " + sys.argv[0] + " <rseed>"
        sys.exit(1)
    random.seed(int(sys.argv[1]))
    # Number of bins
    nbr_bins = 100
    #for dups in [7, 8, 9, 10, 11, 12, 13, 14]:
    for dups in [7]:
        f = open("proportions-" +str(nbr_bins) + "-" + str(dups) + ".dat", "w")
        #for m in range(1, 51):
        for m in range(1, 3):
            mut = float(m) / 100
            print dups, "duplications,", mut, "mutation rate."
            # Build bins with count = 0
            bins = [0]*nbr_bins
            #total_genomes = 1000
            total_genomes = 100
            while sum(bins) != total_genomes:
                model = BanzhafModel()
                sig = [random.randint(0,1) for i in range(0,32)]
                model.setInitialSig(sig)
                model.dm(dups, mut)
                model.addPromoters([["00000000", "TF"], ["11111111", "P"]])
                if(len(model.genes)):
                    fraction = len([1 for x in model.genes if x.type == "TF"]) / float(len(model.genes))
                    index = int(fraction * nbr_bins)
                    if index >= len(bins):
                        index = -1 # if 1.0, place it on last bin
                    bins[index] += 1
            for i in range(0, len(bins)):
                f.write(str(i / float(nbr_bins)) + "\t" + str(bins[i] / float(total_genomes)) + "\t" + str(mut) + "\n")
            f.write("\n") # Newline separating mutation blocks, to help pm3d in gnuplot
        f.close()

if __name__ == "__main__":
    main()
