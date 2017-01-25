#!/usr/bin/env python

import sys # for argv
import random # for random()

from BanzhafModel import *

def main():
    """
    Test program for Banzhaf model
    """
    # Manage arguments
    if(len(sys.argv) != 2):
        print "Usage: " + sys.argv[0] + " <rseed>"
        sys.exit(1)
    random.seed(int(sys.argv[1]))


    for i in range(0, 1):
        model = BanzhafModel()
        sig = [random.randint(0,1) for i in range(0,32)]
        model.setInitialSig(sig)
        model.dm(8, 0.02)
        model.addPromoters([["00000000", "TF"],["11111111", "P"]])
        #model.buildGenes()
        print "="*64

if __name__ == "__main__":
    main()
