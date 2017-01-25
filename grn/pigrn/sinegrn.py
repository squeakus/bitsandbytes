"""
Evolving GRNS to buy and sell shares based on compustat data
"""
import copy, numpy, math
import cgrn as grn
#import grn

def run_grn(genome, delta):
    syncsize = 1000
    stabiter = 10000 / delta
    runiter = syncsize / delta
    completed = False
    results = None
    bh_total = 0

    #run until delta stays above zero
    while not completed:
        regnet = grn.GRN(genome, delta)
        regnet.build_genes()
        regnet.add_extra("EXTRA_sineval", 0.0, [0]*32)
        results = [0] * len(regnet.genes)
        regnet.precalc_matrix()

        #build list of p-genes, quit if only 1 p gene
        p_genes = [ idx for idx, gn in enumerate(regnet.genes) 
                   if gn.gene_type == "P"]    
        if len(p_genes) < 2: return [0],0
        regnet.regulate_matrix(stabiter) # stabilise the grn

        # generate complete sine revolution 
        sineloop =[(math.sin(i) + 1)/2
                      for i in numpy.arange(0,2*math.pi, 0.1)]

        #use two sine waves as input
        sinesamples = sineloop + sineloop
        for inputval in sinesamples:
            extra_vals = {'sineval':inputval}
            regnet.set_extras(extra_vals)
            regnet.regulate_matrix(runiter)

        #only finish if it doesn't go below zero
        if regnet.below_zero == False:
            completed = True
        else:
            delta = delta - 1
            stabiter = 10000 / delta
            runiter = syncsize / delta

    # calculate each p-gene fitness
    offset = 0
    for p_gene in p_genes:
        fitness = 0
        for index, sample in enumerate(sinesamples):
            if index >= offset:
                #unscaled
                target = sinesamples[index-offset] * 0.4
                
                signal = regnet.conc_list[p_gene][index]
                fitness += 1 - abs(target - signal)
        results[p_gene] += fitness

    return results, regnet.conc_list

if __name__ == "__main__":
    import random
    genome = [random.randint(0, 1) for _ in range(0,5000)]
    run_grn(genome, 1)
