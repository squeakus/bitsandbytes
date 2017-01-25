"""
Gene Regulatory Model based on the work of Wolfgang Banzhaf
and Miguel Nicolau.
"""
#threshold Beta and Delta

#normalise the TF and P genes separately

import random, math, graph, sys
from banzhaf_parser import reorder_gene
import numpy as np    # for matrices

class Gene:
    """
    Genes objects for GRN model. Generates protein and stores
    concentration
    """
    def __init__(self, info, gene_type):
        self.promoter =  info[0:32]
        self.enhancer = info[32:64]
        self.inhibitor = info[64:96]
        self.geneinfo = info[96:256] 
        self.protein = []
        self.enh_matrix = []
        self.inh_matrix = []
        self.create_protein()
        self.concentration = 0.0
        self.gene_type = gene_type

    def create_protein(self):
        """Use geneinfo and majority rule to make protein"""
        blocks = len(self.geneinfo) / 32

        for bit in range(32):
            values = []

            for block in range(0, blocks):
                values.append(self.geneinfo[bit + block * 32])
            values.sort() #use midpoint of sort to find majority
            self.protein.append(values[len(values)/2])

class GRN:
    """ Simplified GRN model"""
    def __init__(self):
        """Default constructor, randomly initialised genome"""
        self.genome_size = 5000
        self.sensible_parsing = False
        self.genome = self.random_init()
        self.promoters = [[[0]*8, "TF"], [[1]*8, "P"]]
        self.rest_delta = 0.00005
        self.delta = 1.0
        self.genes = []
        self.change_rate = 0
        self.tf_genes, self.p_genes = 0, 0
        self.conc_list = []

    def random_init(self):
        """randomly populates a genome with ones and zeros"""
        indiv = [random.randint(0, 1) for _ in range(0, self.genome_size)]
        return indiv

    def read_genome(self, filename):
        """parse genome from file"""
        gene_file = open(filename, "r")
        self.genome = eval(gene_file.readline())

    def build_genes(self):
        """Search genome for promoter sites and create gene list"""
        index = 0
        max_length = len(self.genome) - 256

        if not self.sensible_parsing:
            index = 96 - len(self.promoters[0][0])
            max_length = len(self.genome) - (160 + len(self.promoters[0][0]))
            
        while index <= max_length:
            #print "checking:", index
            for seq in self.promoters:
                found = self.genome[index:index+len(seq[0])] == seq[0]
                
                if found:
                    print "found", seq[1], "gene at:", index - 24
                    if seq[1] == "TF": self.tf_genes += 1
                    elif seq[1] == "P": self.p_genes += 1
                
                    if self.sensible_parsing:
                        gene_segment = self.genome[index:index + 256]
                    else:
                        gene_segment = reorder_gene(index,
                                                    seq[0],
                                                    self.genome)
                    self.conc_list.append([])
                    self.genes.append(Gene( gene_segment, 
                                           seq[1]))         
                    index += 255
            index += 1
        self.update_concentrations(True)

    def add_gene(self, concentration, gene_type):
        """Add a gene of specified type and concentration"""
        segment = [random.randint(0,1) for _ in range(256)]
        extra_gene = Gene(segment, gene_type)
        extra_gene.concentration = concentration
        self.genes.append(extra_gene)
        self.conc_list.append([concentration])

    def change_extra(self, val):
        for gene in self.genes:
            if gene.gene_type == 'EXTRA':
                gene.concentration += val

    def precalc_matrix(self):
        """Generate concentration matrix using interdependent TF proteins
        XOR the protein with the enhancer and inhibitor sites on every the gene
        """
        self.enh_matrix = np.zeros((len(self.genes), len(self.genes)))
        self.inh_matrix = np.zeros((len(self.genes), len(self.genes)))

        #iterate proteins and genes to create matrix
        for idx, gene in enumerate(self.genes):            
            if gene.gene_type in ("TF", "EXTRA"):
                protein = gene.protein

                for idy, target in enumerate(self.genes):
                    enhancer, inhibitor = target.enhancer, target.inhibitor

                    xor_enhance = sum(protein[i] != enhancer[i]
                                      for i in range(len(protein)))
                    xor_inhibit = sum(protein[i] != inhibitor[i]
                                      for i in range(len(protein)))

                    self.enh_matrix[idx][idy] = xor_enhance
                    self.inh_matrix[idx][idy] = xor_inhibit
        
        #ensure it is always a negative number
        max_observed =  max(np.max(self.enh_matrix), np.max(self.inh_matrix))
        self.enh_matrix = self.enh_matrix - max_observed
        self.inh_matrix = self.inh_matrix - max_observed

        # precalculate the exponential function
        vector_exp = np.vectorize(math.exp) 
        self.enh_matrix = vector_exp(self.enh_matrix)
        self.inh_matrix = vector_exp(self.inh_matrix)

    def regulate_matrix(self, iterations, stabilising):
        """ Iterate a concentration change"""
        for itr in range(iterations):
            concentrations = []

            for idy, gene in enumerate(self.genes):
                enhance, inhibit, signal = 0, 0, 0

                for idx in range(0, len(self.enh_matrix)):
                    enhance_sig = self.enh_matrix[idx][idy]
                    inhibit_sig = self.inh_matrix[idx][idy]

                    enhance += self.genes[idx].concentration * enhance_sig
                    inhibit += self.genes[idx].concentration * inhibit_sig

                # scale by number of genes
                if gene.gene_type == "EXTRA":
                    print gene.concentration
                elif gene.gene_type == "P":
                    enhance = enhance / self.p_genes
                    inhibit = inhibit / self.p_genes
                    signal = enhance - inhibit
                    gene.concentration += self.delta * signal  
                else:
                    enhance = enhance / self.tf_genes
                    inhibit = inhibit / self.tf_genes
                    signal = enhance - inhibit
                    gene.concentration += self.delta * (signal * gene.concentration)
                # make sure it does not go below zero
                if gene.concentration < 0:
                    gene.concentration = 1e-10
                concentrations.append(gene.concentration)

            self.update_concentrations(False)

            if stabilising:
                if self.change_rate < self.rest_delta:
                    print "breaking early", itr
                    break

        current_concentrations = [i.concentration for i in self.genes]
        return current_concentrations



    def update_concentrations(self, initialising):
        """Normalise the values of the TF and P proteins and add them
        to the concentration list"""
        if initialising:            
            for gene in self.genes:
                if gene.gene_type == "TF":
                    gene.concentration = 1.0/self.tf_genes
                elif gene.gene_type == "P":
                    gene.concentration = 1.0/self.p_genes
 
        tf_total = sum([i.concentration for i in self.genes 
                        if i.gene_type == "TF"])
        p_total = sum([i.concentration for i in self.genes 
                       if i.gene_type == "P"])

        # normalise concentrations separately
        for idx, gene in enumerate(self.genes):
            if gene.gene_type == "TF":
                gene.concentration = gene.concentration / tf_total
            elif gene.gene_type == "P":
                gene.concentration = gene.concentration / p_total

        # add concentrations to results list and calculate change rate
        for idx, gene in enumerate(self.genes):
            if not initialising:
                self.change_rate += abs(self.conc_list[idx][-1]
                                        - gene.concentration)
            self.conc_list[idx].append(gene.concentration)



def main():

    grn = GRN()
    grn.read_genome("eoinseed1.txt")
    grn.build_genes()
    #grn.add_gene(0.0, "EXTRA")
    grn.precalc_matrix()
    grn.regulate_matrix(10000, False)
    # for i in range(0,3):
    #     init_concs = []
    #     for conc in grn.conc_list:
    #         init_concs.append(conc[i])
    #     print "ROUND ", i
    #     print init_concs
    graph.plot_2d(grn.conc_list,0)
    
    #graph.continuous_plot(1000, grn)

if __name__ == "__main__":
    main()
