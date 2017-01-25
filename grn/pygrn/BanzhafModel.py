# Hamming matrix and regulatory matrix
# whats this family malarky about?
# plenty going on in sum function with for loop, adding a boolean?
# shortening the promotor site shortens genome
# promoter length defines search space, 
# calculating the dmevent for the genome? which class it is?
# dmEvent where gene was discovered
#importance of initial bit
#need a protein?
# geneReg why turn off enhance and inhibit?

import random           # for random()
import math             # for log()
from numpy import zeros # for matrices

families = ["Triangle", "Quadrangle", "Pentagon",
        "Hexagon", "Heptagon", "Octagon", "Nonagon", "Decagon",
        "Hendecagon", "Dodecagon", "Triskaidecagon", "Tetrakaidecagon",
        "Pentakaidecagon", "Hexakaidecagon", "Heptakaidecagon",
        "Octakaidecagon", "Enneakaidecagon", "Icosagon", "Icosikaihenagon",
        "Icosikaidigon", "Icosikaitrigon", "Icosikaitetragon",
        "Icosikaipentagon", "Icosikaihexagon", "Icosikaiheptagon",
        "Icosikaioctagon", "Icosikaienneagon", "triacontagon",
        "triacontakaihenagon", "triacontakaidigon", "triacontakaitrigon",
        "triacontakaitetragon", "triacontakaipentagon",
        "triacontakaihexagon", "triacontakaiheptagon",
        "triacontakaioctagon", "triacontakaienneagon", "tetracontagon"]

class Gene:
    """ 
    Genes for Banzhaf model
    Contains protein part and arrays for gene components
    """

    def setGeneInfo(self, geneInfo):
        """
        Called by initialiser, uses bitstring to initialise protein. 
        Checks info is % 32, creates blocks and appends to protein
        """
        newGeneInfo = [int(a) for a in geneInfo]
        if len(newGeneInfo) % 32 != 0:
            raise Exception("Length of gene info (" + str(len(genInfo)) + " is not multiple of 32.")
        self.geneInfo = newGeneInfo
        blocks = len(newGeneInfo) / 32
        # Create protein with majority rule
        self.protein = []
        for bit in range(0, 32):
            values = []
            for block in range(0, blocks):
                values.append(newGeneInfo[bit + block * 32])
            values.sort()
            self.protein.append(values[len(values)/2])

    def __init__(self, loc, enh, inh, prom, type, gInfo, initialBit, dmEvent):
        """Constructor with all info"""
        self.location = loc
        self.enhancer = [int(a) for a in enh]
        self.inhibitor = [int(a) for a in inh]
        self.promoter = [int(a) for a in prom]
        self.type = type
        self.geneinfo = []
        self.protein = []
        self.setGeneInfo(gInfo)
        self.initialBit = initialBit
        self.dmEvent = dmEvent

class BanzhafModel:
    """ Banzhaf Model"""

    def __init__(self):
        """Default constructor, empty genome"""
        self.genome = []
        self.initialSig = []
        self.promoters = []
        self.genes = []
        self.dmEvents = 0
        self.dmProb = 0.0

    def setInitialSig(self, sig):
        """
        Casts bitstring to ints and sets it as initialsig
        This is then used by dup/mut to create genome
        """
        self.initialSig = [int(a) for a in sig]

    def dm(self, dmEvents, dmProb):
        """
        Duplication and mutation initialisation (as opposed to random)
        Given a bit string, append to itself with small mutation
        dmEvents is the number of duplications that are made
        dmProb is the probability of mutation
        """
        self.dmEvents = dmEvents
        
        print "dmevents", self.dmEvents
        self.dmProb = dmProb

        if self.dmProb < 0.0 or self.dmProb > 1.0:
            raise Exception("DM mutation (" + str(self.dmProb) + ") is not between 0.0 and 1.0.")
        self.genome = [x for x in self.initialSig]

        for dm in range(0, self.dmEvents):
            newSig = [bit if random.random() > self.dmProb else (bit + 1) % 2 for bit in self.genome]
            self.genome.extend(newSig)

        print "genome length:", len(self.genome)

    def addPromoters(self, promoters):
        """
        Add promoter ID and type to promoter list.
        Once complete build the genes
        """
        for promoter in promoters:
            intPromoter = [int(a) for a in promoter[0]]
            if len(intPromoter) > 32:
                raise Exception("Promoters have a max size of 32bits.")
            self.promoters.append((intPromoter, promoter[1]))
        self.buildGenes()


    def buildGenes(self):
        """
        chops up the genome into genes. Each gene consists of an enhancer,
        inhibitor, promoter(ID), type, info, initial bit, and dmevent
        """
        print "promoters:", len(self.promoters),"genes:",len(self.genome)/256
        if len(self.genome) == 0:
            return
        self.genes = [] # Clean current list of genes
        index = 0
        foundGenes = 0
        # Search genes up to 256 bits from the end (genes are 256 bits long)
        print "no of promoters:", len(self.promoters)
        while index < len(self.genome) - 256:
            # Cycle through promoter sequences
            for promoterSeq in self.promoters: 
                found = self.genome[index + 96 - len(promoterSeq[0]):index + 96] == promoterSeq[0]
                # If promoter sequence found on promoter site location
                if found:
                    print "found!", self.genome[index + 96 - len(promoterSeq[0]):index + 96]
                    foundGenes += 1
                    #what in gods name is going on here?
                    dmEvent = (index + 256) / len(self.initialSig)
                    if dmEvent != 0:
                        dmEvent = math.floor(math.log(dmEvent, 2) + 1)
                        print "new dmevent", dmEvent
                    self.genes.append(Gene(index,                   # Location
                            self.genome[index + 0 :index + 32],     # Enhancer site
                            self.genome[index + 32:index + 64],     # Inhibitor site
                            self.genome[index + 64:index + 96],     # Promoter site
                            promoterSeq[1],                         # Type
                            self.genome[index + 96:index + 256],    # Gene info
                            index % len(self.initialSig),           # Initial bit
                            dmEvent))                               # dmEvent where gene was discovered
                    newGene = self.genes[-1]

                    # Choose family name and set it
                    usedFamilies = []
                    newGene.family = ""
                    for g in self.genes[:-1]:
                        if g.family not in usedFamilies:
                            usedFamilies.append(g.family)
                        if g.initialBit == newGene.initialBit:
                            newGene.family = g.family
                            break
                    
                    # New family of genes
                    if newGene.family == "":
                        for shape in families:
                            if shape not in usedFamilies:
                                newGene.family = shape
                                break
                    index += 255 # Add 255 as index is increased at end of while loop
                    break
            index += 1
        print "found genes", foundGenes

    def regulatoryMatrix(self, enhancing, inhibiting, threshold = None):
        """
        Generate matrix of concentrations for the interdependent TF genes
        """
        rMatrix = zeros((len(self.genes), len(self.genes)))
        for tf in range(0, len(self.genes)):
            
            #if transcription factor gene (regular gene)
            if self.genes[tf].type == "TF":
                for target in range(0, len(self.genes)):
                    if(enhancing):
                        eSignal = sum(self.genes[tf].protein[i] != self.genes[target].enhancer[i]
                                for i in range(0, len(self.genes[tf].protein)))
                    if(inhibiting):
                        iSignal = sum(self.genes[tf].protein[i] != self.genes[target].inhibitor[i]
                                      for i in range(0, len(self.genes[tf].protein)))
                    signal = 0
                    if enhancing and inhibiting:
                        signal = eSignal - iSignal
                    elif enhancing:
                        signal = eSignal
                    elif inhibiting:
                        signal = iSignal
                    if not threshold or signal >= threshold:
                        rMatrix[tf][target] = signal
        return rMatrix

    def hammingMatrix(self):
        """ 
        sums the whatnow?
        """
        print "calling hamming"
        hMatrix = zeros((len(self.genes), len(self.genes)))
        for g1 in range(0, len(self.genes)):
            for g2 in range(0, len(self.genes)):
                hMatrix[g1][g2] = sum(self.genes[g1].enhancer[i] != self.genes[g2].enhancer[i]
                                        for i in range(0, len(self.genes[g1].enhancer)))
                hMatrix[g1][g2] += sum(self.genes[g1].inhibitor[i] != self.genes[g2].inhibitor[i]
                                                                for i in range(0, len(self.genes[g1].inhibitor)))
                hMatrix[g1][g2] += sum(self.genes[g1].promoter[i] != self.genes[g2].promoter[i]
                                                                for i in range(0, len(self.genes[g1].promoter)))
                hMatrix[g1][g2] += sum(self.genes[g1].geneInfo[i] != self.genes[g2].geneInfo[i]
                                        for i in range(0, len(self.genes[g1].geneInfo)))
        return hMatrix

    def printRegulatoryGdf(self, filename, added, enhancing, inhibiting, threshold):
        """ 
        Prints out gdf file for Gephi graphs
        added -> use links with added regulatory strength (and ignore next 2 options)
        enhancing -> draw enhancing links
        inhibiting -> draw inhibiting links
        """
        # First build matrices, to build list of connected nodes
        connectedNodes = []
        if added:
            aMatrix = self.regulatoryMatrix(True, True, threshold)
            for i in range(len(aMatrix)):
                if aMatrix[i].any() or aMatrix[:,i].any():
                    connectedNodes.append(i)
        else:
            if enhancing:
                eMatrix = self.regulatoryMatrix(True, False, threshold)
                for i in range(len(eMatrix)):
                    if eMatrix[i].any() or eMatrix[:,i].any():
                        connectedNodes.append(i)
            if inhibiting:
                iMatrix = self.regulatoryMatrix(False, True, threshold)
                for i in range(len(iMatrix)):
                    if i not in connectedNodes and (iMatrix[i].any() or iMatrix[:,i].any()):
                        connectedNodes.append(i)
        f = open(filename, "w")
        # Nodes
        f.write("nodedef> name VARCHAR, order NUMERIC, location NUMERIC, \
                enhancer VARCHAR, inhibitor VARCHAR, promoter VARCHAR, \
                type VARCHAR, geneinfo VARCHAR, initialbit NUMERIC, \
                dmevent NUMERIC, family VARCHAR, function VARCHAR, color VARCHAR\n")
        for i in range(len(self.genes)):
            if i in connectedNodes:
                f.write("G" + str(i + 1))                               # name
                f.write("," + str(i + 1))                               # order
                f.write("," + str(self.genes[i].location))              # location
                f.write(",")
                for bit in self.genes[i].enhancer: f.write(str(bit))    # enhancer
                f.write(",")
                for bit in self.genes[i].inhibitor: f.write(str(bit))   # enhancer
                f.write(",")
                for bit in self.genes[i].promoter: f.write(str(bit))    # enhancer
                f.write("," + self.genes[i].type)                       # type
                f.write(",")
                for bit in self.genes[i].geneInfo: f.write(str(bit))    # gene info
                f.write("," + str(self.genes[i].initialBit))            # initial bit
                f.write("," + str(self.genes[i].dmEvent))               # dm event
                f.write("," + self.genes[i].family)                     # family
                f.write("," + self.genes[i].type)                       # function (same as type)
                if self.genes[i].type == "TF":
                    f.write(",'255,0,0'")
                elif self.genes[i].type == "P":
                    f.write(",'0,255,0'")
                else:
                    f.write(",'128,128,128'")
                f.write("\n")
        # Edges
        f.write("edgedef> node1 VARCHAR, node2 VARCHAR, directed BOOLEAN, \
                enhancing BOOLEAN, inhibiting BOOLEAN, weight NUMERIC, \
                color VARCHAR\n")
        if added:
            for row in range(len(aMatrix)):
                for col in range(len(aMatrix[row])):
                    if not threshold or aMatrix[row][col] >= threshold:
                        f.write("G" + str(row + 1) + ",G" + str(col + 1) + ",true,false,false," + str(aMatrix[row][col]) + ",'0,0,0'\n")
        else:
            if enhancing:
                for row in range(len(eMatrix)):
                    for col in range(len(eMatrix[row])):
                        if not threshold or eMatrix[row][col] >= threshold:
                            f.write("G" + str(row + 1) + ",G" + str(col + 1) + ",true,false,false," + str(eMatrix[row][col]) + ",'0,0,0'\n")
            if inhibiting:
                for row in range(len(iMatrix)):
                    for col in range(len(iMatrix[row])):
                        if not threshold or iMatrix[row][col] >= threshold:
                            f.write("G" + str(row + 1) + ",G" + str(col + 1) + ",true,false,false," + str(iMatrix[row][col]) + ",'0,0,0'\n")
        f.close()

        
    def printRegulatoryDot(self, filename, added, enhancing, inhibiting, threshold):
        """ 
        Prints out network to a dot file.
        added -> use links with added regulatory strength
        enhancing -> draw enhancing links
        inhibiting -> draw inhibiting links
        """
        # First build matrices, to build list of connected nodes
        connectedNodes = []
        if added:
            aMatrix = self.regulatoryMatrix(True, True, threshold)
            for i in range(len(aMatrix)):
                if aMatrix[i].any() or aMatrix[:,i].any():
                    connectedNodes.append(i)
        else:
            if enhancing:
                eMatrix = self.regulatoryMatrix(True, False, threshold)
                for i in range(len(eMatrix)):
                    if eMatrix[i].any() or eMatrix[:,i].any():
                        connectedNodes.append(i)
            if inhibiting:
                iMatrix = self.regulatoryMatrix(False, True, threshold)
                for i in range(len(iMatrix)):
                    if i not in connectedNodes and (iMatrix[i].any() or iMatrix[:,i].any()):
                        connectedNodes.append(i)
        f = open(filename, "w")
        f.write("digraph Banzhaf {\n" + "\tratio=\".7\"\n" + "\tsize=\"59.4,42\"\n")
        # Nodes
        for i in range(len(self.genes)):
            if i in connectedNodes:
                f.write("\tG" + str(i + 1))                             # name
                if self.genes[i].type == "TF":
                    f.write(" [shape = octagon];\n")
                elif self.genes[i].type == "P":
                    f.write(" [shape = doubleoctagon];\n")
                else:
                    f.write(" [shape = none];\n")
        # Edges
        if added:
            for row in range(len(aMatrix)):
                for col in range(len(aMatrix[row])):
                    if not threshold or aMatrix[row][col] >= threshold:
                        f.write("\tG" + str(row + 1) + "->G" + str(col + 1) + "[weight=" + str(rMatrix[row][col]) + "];\n")
        else:
            if enhancing:
                for row in range(len(eMatrix)):
                    for col in range(len(eMatrix[row])):
                        if not threshold or eMatrix[row][col] >= threshold:
                            f.write("\tG" + str(row + 1) + "->G" + str(col + 1) + "[weight=" + str(eMatrix[row][col]) + "];\n")
            if inhibiting:
                for row in range(len(iMatrix)):
                    for col in range(len(iMatrix[row])):
                        if not threshold or iMatrix[row][col] >= threshold:
                            f.write("\tG" + str(row + 1) + "->G" + str(col + 1) + "[weight=" + str(iMatrix[row][col]) + "][style=dotted];\n")
        f.write("}\n")
        f.close()
