"""
Gene Regulatory Model based on the work of Wolfgang Banzhaf
and Miguel Nicolau.
"""
# Bug when casting from ctype array to nparray
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import random, math, graph, sys
from banzhaf_parser import reorder_gene
import numpy as np    # for matrices
import ctypes as ct   # for grn c-library

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
    def __init__(self, genome=None, delta=1):
        """Default constructor, randomly initialised genome"""
        if genome == None:
            self.genome = self.random_init(5000)
        else:
            self.genome = genome
            
        ct.cdll.LoadLibrary('libgrn.so')
        self.grnlib = ct.CDLL('libgrn.so')
        self.sensible_parsing = False
        self.promoters = [[[0]*8, "TF"], [[1]*8, "P"]]
        self.delta = delta
        self.below_zero = False
        self.extra_limit = 0.4        
        self.tf_genes = []
        self.p_genes = []
        self.extras = 0
        self.genes = []
        self.conc_list = []
        self.weight_matrix = []

    def random_init(self, genome_size):
        """randomly populates a genome with ones and zeros"""
        indiv = [random.randint(0, 1) for _ in range(0, genome_size)]
        return indiv

    def read_genome(self, filename):
        """parse genome list from file"""
        gene_file = open(filename, "r")
        self.genome = eval(gene_file.readline())

    def build_genes(self):
        """Search genome for promoter sites and create gene list"""
        index = 0
        gene_cnt = 0
        max_length = len(self.genome) - 256
        
        if not self.sensible_parsing:
            index = 96 - len(self.promoters[0][0])
            max_length = len(self.genome) - (160 + len(self.promoters[0][0]))
            
        while index <= max_length:
            for seq in self.promoters:
                found = self.genome[index:index+len(seq[0])] == seq[0]
                
                if found and index <= max_length:
                    if seq[1] == "TF": 
                        self.tf_genes.append(gene_cnt)
                    elif seq[1] == "P": 
                        self.p_genes.append(gene_cnt)
                        
                    if self.sensible_parsing:
                        gene_segment = self.genome[index:index + 256]
                    else:
                        gene_segment = reorder_gene(index,
                                                    seq[0],
                                                    self.genome)
                    self.genes.append(Gene( gene_segment, 
                                           seq[1]))
                    index += 255
                    gene_cnt += 1
            index += 1

        # once genes have been counted, set initial concentration
        self.conc_list = []
        for gene in self.genes:
            self.conc_list.append([])
            if gene.gene_type == "TF":
                gene.concentration = 1.0/len(self.tf_genes)
            elif gene.gene_type == "P":
                gene.concentration = 1.0/len(self.p_genes)

    def add_extra(self, gene_type, concentration, signature=None):
        """Initialise and add a gene"""
        self.extras +=1
        segment = [random.randint(0, 1) for _ in range(256)]
        extra_gene = Gene(segment, gene_type)
        extra_gene.concentration = concentration
        
        if not signature == None:
            extra_gene.protein = signature
            
        self.genes.append(extra_gene)
        self.conc_list.append([])
        
    def set_extras(self, extra_vals):
        """Set all EXTRA protein concs"""
        e_total = sum([ extra_vals[key] for key in extra_vals])
        for key in extra_vals:
            for gene in self.genes:
                extra_name = 'EXTRA_'+key
                if gene.gene_type == extra_name:
                    if len(extra_vals) == 1:
                        gene.concentration = extra_vals[key]*self.extra_limit
                    else:
                        gene.concentration = ((extra_vals[key] / e_total)
                                              * self.extra_limit)

    def precalc_matrix(self):
        """Generate concentration matrix using interdependent TF and
        input EXTRA proteins, XOR the protein with the enhancer and
        inhibitor sites on every the gene and calculate exponential.
        Then generate weights by subtracting inhibitor from the enhancer
        """
        enh_matrix = np.zeros((len(self.genes),
                               len(self.genes)-self.extras))
        inh_matrix = np.zeros((len(self.genes),
                               len(self.genes)-self.extras))

        #iterate proteins and genes to create matrix
        for idx, gene in enumerate(self.genes):
            if not gene.gene_type == "P":
                protein = gene.protein

                for idy in range(len(self.genes)-self.extras):
                    target = self.genes[idy]
                    enhancer, inhibitor = target.enhancer, target.inhibitor
                    
                    xor_enhance = sum(protein[i] != enhancer[i]
                                      for i in range(len(protein)))
                    xor_inhibit = sum(protein[i] != inhibitor[i]
                                      for i in range(len(protein)))
                
                    enh_matrix[idx][idy] = xor_enhance
                    inh_matrix[idx][idy] = xor_inhibit

        #ensure it is always a negative number
        max_observed =  max(np.max(enh_matrix), np.max(inh_matrix))
        enh_matrix = enh_matrix - max_observed
        inh_matrix = inh_matrix - max_observed

        # precalculate the exponential function and generate weight matrix
        vector_exp = np.vectorize(math.exp) 
        enh_matrix = vector_exp(enh_matrix)
        inh_matrix = vector_exp(inh_matrix)
        self.weight_matrix = enh_matrix - inh_matrix
        
    def regulate_matrix(self, iterations):
        """ Pass weight array, concs, tfs and p genes to the 
        grnlib for fast calculation"""
        weights = self.weight_matrix.astype(np.float64)
        cweights = weights.ctypes.data_as(ct.POINTER(ct.c_double))
        rows, cols = self.weight_matrix.shape

        concs = []
        for gene in self.genes:
            concs.append(gene.concentration);
        e_total = ct.c_double(sum([i.concentration for i in self.genes 
                                   if i.gene_type.startswith('EXTRA')]))
        # compute array lengths
        conc_len = len(concs)
        tf_len = len(self.tf_genes)
        p_len = len(self.p_genes)

        # cast to c arrays
        cconcs = (ct.c_double * conc_len)(*concs)
        ctfs = (ct.c_int * tf_len)(*self.tf_genes)
        cps = (ct.c_int * p_len)(*self.p_genes)

        # call grnlib
        for itr in range(iterations):
            bzero = self.grnlib.regulate(conc_len, cconcs,
                                         rows, cols, cweights,
                                         tf_len, ctfs, p_len, cps,
                                         self.delta, e_total)
            if bzero == 1:
                self.below_zero = True

        # cast ctypes to numpy and set gene values
        final_concs = np.ctypeslib.as_array(
            (ct.c_double * conc_len).from_address(ct.addressof(cconcs)))

        for idx, gene in enumerate(self.genes):
            gene.concentration = final_concs[idx]
            self.conc_list[idx].append(final_concs[idx])
        
        current_concentrations = np.array([i.concentration 
                                           for i in self.genes])

        return current_concentrations, itr + 1

def main():
    #speed and comparison code
    import time
    
    start_time = time.time()
    for seed in range(10):
        random.seed(seed)
        stabiter = 10000
        runiter = 1000
        grn = GRN(delta=1)

        grn.build_genes()    
        grn.add_extra("EXTRA_sineval", 0.0, [0]*32)
        grn.precalc_matrix()
        grn.regulate_matrix(stabiter)

        onff = 0
        for i in range(0, 50):
            if i % 10 == 0:
                if onff == 1:
                    onff = 0
                else:
                    onff = 1

            inputval = onff
            extra_vals = {'sineval': inputval}
            grn.set_extras(extra_vals)
            grn.regulate_matrix(runiter)

        for conc in grn.conc_list:
            print conc[-1]

        filename = "test"+str(seed)
        graph.plot_2d(grn.conc_list, filename)
    print "took", str(time.time() - start_time)

if __name__ == "__main__":
    main()
