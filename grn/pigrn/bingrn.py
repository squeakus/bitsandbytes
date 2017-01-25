"""
Evolving GRNS to buy and sell shares based on compustat data
"""

import sys, random, grn, popstrat, graph, pp, copy, numpy, math

def run_grn(genome, delta):
    stabiter = 10000 / delta
    runiter = 1000 / delta
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

        # use sine as input value for grn
        onff = 0
        for i in range(0, 50):
            if i % 10 == 0:
                if onff == 1:
                    onff = 0
                else:
                    onff = 1                        
            inputval = onff
            extra_vals = {'sineval':inputval}
            regnet.set_extras(extra_vals)
            regnet.regulate_matrix(runiter)
        samples = range(stabiter, (runiter*50)+stabiter, runiter)

        #only finish if it doesn't go below zero
        if regnet.below_zero == False:
            completed = True
        else:
            delta = delta - 1
            stabiter = 10000 / delta
            runiter = 1000 / delta

    # read results from grn to calc fitness
    for p_gene in p_genes:
        fitness = 0
        onff = 0 
        for j in range(0, 50):
            # if j % 10 == 0:
            #     if onff == 1:
            #         onff = 0
            #     else:
            #         onff = 1
            # target = onff 
            idx, idy = p_gene, samples[int(j)]
            target = regnet.conc_list[-1][idy]
            signal = regnet.conc_list[idx][idy]
            # print "T", target, "S",regnet.conc_list[-1][idy]
            fitness += 1 - abs(target - signal)
        results[p_gene] += fitness
    return results, regnet.conc_list

def singlecore():
    delta = 20    
    #set up the evo strategy
    best_list, mut_list = [], []
    evo = popstrat.Evostrategy(5000,50)
    children = evo.iterate(evo.pop)

    for i in range(50):
        for child in children:
            start_time = time.time()
            results, conclist = run_grn(child['genome'], delta)
            bestidx = results.index(max(results))
            child['fitness'] = results[bestidx]
            print "fitness:",child['fitness']
            
        children = evo.iterate(children)
        bestgenome = evo.pop[-1]['genome']
        results, conclist = run_grn(bestgenome, delta)
        filename = "best_gen_"+str(i)
        graph.plot_2d(conclist, filename)
    
        if evo.adaptive:
            evo.adapt_mutation()

        best_list.append(evo.pop[-1]['fitness'])
        mut_list.append(evo.mut_rate)
        
    print "best overall fitness", evo.pop[-1]['fitness']
    
    graph.plot_2d([best_list], 'bestfit')
    graph.plot_2d([mut_list], 'mutrate')


def multicore():
    delta = 20
    #set up the evo strategy
    best_list, mut_list = [], []
    evo = popstrat.Evostrategy(5000, 50)
    children = evo.iterate(evo.pop)

    nodes = ("*",)
    job_server = pp.Server(8, ppservers=nodes)
    print "Starting pp with", job_server.get_ncpus(), "workers"

    start_time = time.time()

    for i in range(50):
        run_time = time.time()
        jobs = [(child, job_server.submit(run_grn, 
                                          (child['genome'], 
                                           delta),
                                           (),
                                           ("grn","numpy","math")))
                                           for child in children]
        for child, result in jobs:
            results, conclist = result()
            bestidx = results.index(max(results))
            child['fitness'] = results[bestidx]

        #plotting the best with colors
        children = evo.iterate(children)
        bestgenome = evo.pop[-1]['genome']
        bestresult, conclist = run_grn(bestgenome, delta)
        bestidx = bestresult.index(max(bestresult))
        filename = "best_gen_"+str("%03d" % i)
        print filename

        colors = []
        simplist = []
        for idx, result in enumerate(bestresult):
            if idx == len(bestresult)-1:
                simplist.append(conclist[idx])
                colors.append('k')
            elif idx == bestidx:
                colors.append('g')
                simplist.append(conclist[idx])
            # elif result == 0:
            #     colors.append('b')
            # else:
            #     colors.append('r')
        graph.plot_2d(simplist, filename, colors)

        print "gen:", evo.gen_count, "fitness:", evo.pop[-1]['fitness']

        if evo.adaptive:
            evo.adapt_mutation()

        best_list.append(evo.pop[-1]['fitness'])
        mut_list.append(evo.mut_rate)

    mutfile = open('mutrate.txt','a')
    mutfile.write(str(mut_list)+'\n')
    mutfile.close()

    
if __name__ == "__main__":
    import time
    if(len(sys.argv) != 2):
        print "Usage: " + sys.argv[0] + " <rseed>"
        sys.exit(1)
    seed = int(sys.argv[1])
    random.seed(seed)
    filename = "results-"+str(seed)+".dat"    
    multicore()
