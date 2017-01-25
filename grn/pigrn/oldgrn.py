"""
Gene Regulatory Model based on the work of Wolfgang Banzhaf
and Miguel Nicolau.
"""
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
    def __init__(self, genome=None, delta=1):
        """Default constructor, randomly initialised genome"""
        if genome == None:
            self.genome = self.random_init(5000)
        else:
            self.genome = genome
        
        self.sensible_parsing = False
        self.promoters = [[[0]*8, "TF"], [[1]*8, "P"]]
        self.delta = delta
        self.below_zero = False
        self.rest_delta = 0.00005 * delta
        self.extra_limit = 0.4
        self.change_rate = 0
        self.tf_genes = 0
        self.p_genes = 0
        self.extras = 0
        self.genes = []
        self.conc_list = []
        self.enh_matrix = []
        self.inh_matrix = []

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
        max_length = len(self.genome) - 256
        
        if not self.sensible_parsing:
            index = 96 - len(self.promoters[0][0])
            max_length = len(self.genome) - (160 + len(self.promoters[0][0]))
            
        while index <= max_length:
            for seq in self.promoters:
                found = self.genome[index:index+len(seq[0])] == seq[0]
                
                if found and index <= max_length:
                    #print "found", seq[1], "gene at:", index - 24
                    if seq[1] == "TF": self.tf_genes += 1
                    elif seq[1] == "P": self.p_genes += 1
                
                    if self.sensible_parsing:
                        gene_segment = self.genome[index:index + 256]
                    else:
                        gene_segment = reorder_gene(index,
                                                    seq[0],
                                                    self.genome)
                    self.genes.append(Gene( gene_segment, 
                                           seq[1]))                 
                    index += 255
            index += 1
        self.update_concentrations(True)

    def update_concentrations(self, initialising):
        """Normalise the values of the TF and P proteins and add them
        to the concentration list"""
        if initialising:
            self.conc_list = []
            for gene in self.genes:
                self.conc_list.append([])
                if gene.gene_type == "TF":
                    gene.concentration = 1.0/self.tf_genes
                elif gene.gene_type == "P":
                    gene.concentration = 1.0/self.p_genes
                elif gene.gene_type.startswith("EXTRA"):
                    gene.concentration = 0
 
        tf_total = sum([i.concentration for i in self.genes 
                        if i.gene_type == "TF" ])
        p_total = sum([i.concentration for i in self.genes 
                       if i.gene_type == "P"])
        e_total = sum([i.concentration for i in self.genes 
                       if i.gene_type.startswith('EXTRA')])
        
        # normalise concentrations separately
        for idx, gene in enumerate(self.genes):
            if gene.gene_type == "TF":
                gene.concentration *= 1.0 - e_total
                gene.concentration = gene.concentration / tf_total
                if gene.concentration < 1e-10:
                    gene.concentration = 1e-10

            elif gene.gene_type == "P":
                gene.concentration = gene.concentration / p_total
                if gene.concentration < 1e-10:
                    gene.concentration = 1e-10

        # add concentrations to results list and calculate change rate
        change_rate = 0
        for idx, gene in enumerate(self.genes):
            if not initialising:
                change_rate += abs(self.conc_list[idx][-1]
                                   - gene.concentration)
            self.conc_list[idx].append(gene.concentration)
        return change_rate

        
    def add_extra(self, gene_type, concentration, signature=None):
        self.extras +=1
        """Initialise and add a gene"""
        segment = [random.randint(0, 1) for _ in range(256)]
        extra_gene = Gene(segment, gene_type)
        extra_gene.concentration = concentration
        
        if not signature == None:
            extra_gene.protein = signature
            
        self.genes.append(extra_gene)
        self.conc_list.append([concentration])
        
    def change_extra(self, name, val):
        """Increment or decrement the concentration 
        of an EXTRA protein"""
        e_total = sum([i.concentration for i in self.genes 
                       if i.gene_type.startswith('EXTRA')])

        #cap amounts of extra protein
        if e_total > self.extra_limit and val > 0:
            val = 0
        for gene in self.genes:
            if gene.gene_type == name:
                gene.concentration += val

    def set_extras(self, extra_vals):
        """Set all EXTRA protein concentrations"""
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
        """
        self.enh_matrix = np.zeros((len(self.genes),
                                    len(self.genes)-self.extras))
        self.inh_matrix = np.zeros((len(self.genes),
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

    def regulate_matrix(self, iterations):
        """ iterate concentration calculations"""
        for itr in range(iterations):
            concentrations = []            
            for idy in range(len(self.genes)-self.extras):
                gene = self.genes[idy]
                enhance, inhibit, signal = 0, 0, 0
                for idx in range(0, len(self.enh_matrix)):
                    enhance_sig = self.enh_matrix[idx][idy]
                    inhibit_sig = self.inh_matrix[idx][idy]

                    enhance += self.genes[idx].concentration * enhance_sig
                    inhibit += self.genes[idx].concentration * inhibit_sig

                # scale by number of proteins, ignore extra concentrations
                enhance = enhance / (self.tf_genes + self.extras)
                inhibit = inhibit / (self.tf_genes + self.extras)
                if gene.gene_type == "P":
                    signal = enhance - inhibit
                    gene.concentration += self.delta * signal

                elif gene.gene_type == "TF":
                    signal = (enhance - inhibit) * gene.concentration
                    gene.concentration += self.delta * signal 
                    if gene.concentration < 0:
                        self.below_zero = True #delta too damn high

                # never goes to zero
                concentrations.append(gene.concentration)

            # normalise and add to the conc array
            change_rate = self.update_concentrations(False)
            
        current_concentrations = [i.concentration for i in self.genes]
        return current_concentrations, itr + 1

def main(): 
    import time
    start_time = time.time()
    for seed in range(10):
        random.seed(seed)
        stabiter = 10000
        runiter = 1000
        grn = GRN(delta=1)

        #grn.read_genome("moo.dat")
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
        filename = "conc"+str(seed)
        graph.plot_2d(grn.conc_list, filename)
    print "took", str(time.time() - start_time)
if __name__ == "__main__":
    main()
