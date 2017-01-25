#!/usr/bin/env python
#randint(0,2)


import sys # for argv
import random # for random()

from BanzhafModel import *

def main():
    """
    Extract various adjacency matrices
    """
    # Manage arguments
    if(len(sys.argv) != 2):
        print "Usage: " + sys.argv[0] + " <rseed>"
        sys.exit(1)
    random.seed(int(sys.argv[1]))
    model = BanzhafModel()
    model.setInitialSig([random.randint(0,2) for i in range(0,32)])
    model.dm(12, 0.02)
    model.addPromoters([["00000000", "TF"], ["11111111", "P"]])
    print [g.type for g in model.genes]
    rMatrix = model.regulatoryMatrix(True, True)
    print "R-Matrix"
    print rMatrix
    hMatrix = model.hammingMatrix()
    print hMatrix
    print [g.type for g in model.genes]

#    for th in (1,11,21,31):
#        model.printRegulatoryDot("graphs/graph-" + str(th) + ".dot", False, True, True, th)
#        model.printRegulatoryGdf("graphs/graph-" + str(th) + ".gdf", False, True, True, th)

if __name__ == "__main__":
    main()
