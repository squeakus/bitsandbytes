import popstrat, pp, time, sys, random, graph
from sinegrn import run_grn

def singlecore(filename, delta, popsize, generations):
    #set up the evo strategy
    best_list, mut_list = [], []
    evo = popstrat.Evostrategy(5000, popsize)
    children = evo.iterate(evo.pop)

    for i in range(generations):
        for child in children:
            start_time = time.time()
            results, conclist = run_grn(child['genome'], delta)
            bestidx = results.index(max(results))
            child['fitness'] = results[bestidx]
            print "fitness:",child['fitness'],"time taken", str(time.time()-start_time)
            
        children = evo.iterate(children)
        bestgenome = evo.pop[-1]['genome']
        results, conclist = run_grn(bestgenome, delta)
        # bestidx = results.index(max(results))
        # fitness = results[bestidx]
        filename = "best_gen_"+str(i)
        graph.plot_2d(conclist, filename)
    
        if evo.adaptive:
            evo.adapt_mutation()

        best_list.append(evo.pop[-1]['fitness'])
        mut_list.append(evo.mut_rate)
        
    print "best overall fitness", evo.pop[-1]['fitness']
    
    graph.plot_2d([best_list], 'bestfit')
    graph.plot_2d([mut_list], 'mutrate')


def multicore(filename, delta, popsize, generations):
    #set up the evo strategy
    evo = popstrat.Evostrategy(5000, popsize)
    children = evo.iterate(evo.pop)

    nodes = ("*",)
    job_server = pp.Server(8, ppservers=nodes)
    print "Starting pp with", job_server.get_ncpus(), "workers"

    start_time = time.time()

    for i in range(generations):
        run_time = time.time()
        jobs = [(child, job_server.submit(run_grn, 
                                          (child['genome'], 
                                           delta),
                                           (),
                                           ("cgrn as grn","numpy","math")))
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

        colors = []
        simplist = []
        for idx, result in enumerate(bestresult):
            if idx == len(bestresult)-1:
                simplist.append(conclist[idx])
                colors.append('k')
            elif idx == bestidx:
                colors.append('g')
                simplist.append(conclist[idx])
            else:
                simplist.append(conclist[idx])
                colors.append('r')
        graph.plot_2d(simplist, filename, colors)
        print "gen:", evo.gen_count, "fitness:", evo.pop[-1]['fitness']

        if evo.adaptive:
            evo.adapt_mutation()

        res_file = open(filename,"a")
        res_file.write(str(evo.pop[-1])+'\n')
        res_file.close()

if __name__ == "__main__":
    import time
    if(len(sys.argv) != 2):
        print "Usage: " + sys.argv[0] + " <rseed>"
        sys.exit(1)
    seed = int(sys.argv[1])
    random.seed(seed)
    filename = "results-"+str(seed)+".dat"    
    multicore(filename, delta=20, popsize=50, generations=50)
