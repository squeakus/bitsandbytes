"""
Evolving GRNS to buy and sell shares based on compustat data
"""

import sys, random, grn, popstrat, csvreader, graph, pp, copy

def run_grn(genome, companies, delta):
    stabiter = 10000 / delta
    runiter = 1000 / delta
    completed = False
    regnet = grn.GRN(genome)
    regnet.build_genes()
    train_results = [0] * regnet.p_genes
    test_results = [0] * regnet.p_genes
    bh_total = 0

    for compid, data in enumerate(companies):
        # run the grn with company inputs
        while not completed:
            cash_total = 1000
            cash_list = []
            signal = []
            buy_hold = 1000 / data['shareprice'][0]

            regnet = grn.GRN(genome, delta)
            regnet.build_genes()
            regnet.add_extra("EXTRA_priceearn", 0.0, [0]*32)
            regnet.add_extra("EXTRA_pricebook", 0.0, [1]*32)
            regnet.add_extra("EXTRA_beta", 0.0, [0]*16 +[1]*16)
            regnet.add_extra("EXTRA_cashflow", 0.0, [1]*16 +[0]*16)
            regnet.precalc_matrix()

            #build list of p-genes, quit if only 1 p gene
            p_genes = [ idx for idx, gn in enumerate(regnet.genes) 
                       if gn.gene_type == "P"]    
            if len(p_genes) < 2: return 0,0

            regnet.regulate_matrix(stabiter) # stabilise the regnet

            # use compustat as REGNET input for 60 months
            for month in range(60):
                extra_vals = {}
                for key in data:
                    if not key == 'shareprice':
                        extra_vals[key] = data[key][month]
                regnet.set_extras(extra_vals)
                regnet.regulate_matrix(runiter)
            samples = range(stabiter, (runiter*60)+stabiter, runiter)

            if regnet.below_zero == False:
                completed = True
            else:
                delta = delta - 1
                stabiter = 10000 / delta
                runiter = 1000 / delta

        #calculate all the profit for the p_genes
        for i, p_gene in enumerate(p_genes):      
            cash_total = 1000    
            share_count = 0
            invest = 0

            for month in range(60):
                share_price = data['shareprice'][month]
                cash_total = (share_price*share_count) + cash_total
                cash_list.append(cash_total)
                idx, idy = p_gene, samples[month]
                signal.append(regnet.conc_list[idx][idy])
                invest = regnet.conc_list[idx][idy] * cash_total
                cash_total = cash_total - invest
                share_count = invest / share_price

            final = (share_price*share_count) + cash_total
            if compid < 5:
                train_results[i] += final
            else:
                test_results[i] += final    
        #calculate buy and hold amount
        buy_hold = buy_hold * share_price
        bh_total += buy_hold

    bestidx = train_results.index(max(train_results)) 
    return train_results[bestidx], test_results[bestidx]

def main():
    import time
    if(len(sys.argv) != 2):
        print "Usage: " + sys.argv[0] + " <rseed>"
        sys.exit(1)
    seed = int(sys.argv[1])
    random.seed(seed)

    delta = 20
    filename = "results-"+str(seed)+".dat"
    
    #read in the compustat data for given companies
    companies = [] 
    company_list = ['ARCHER-DANIELS-MIDLAND CO', 'ARTHUR J GALLAGHER & CO',
                    'ASCENA RETAIL GROUP INC', 'ASTEC INDUSTRIES INC', 
                    'ASTORIA FINANCIAL CORP', 'BLACKBAUD INC', 
                    'BMC SOFTWARE INC','BOSTON BEER INC  -CL A',
                    'BRADY CORP', 'BRIGGS & STRATTON']

    database = csvreader.CSVReader('compustat.csv')
    for company in company_list:
        companies.append(database.get_company(company))
    
    #set up the evo strategy
    best_list, mut_list = [], []
    evo = popstrat.Evostrategy(5000, 50)
    children = evo.iterate(evo.pop)

    nodes = ("*",)
    job_server = pp.Server(0, ppservers=nodes)
    print "Starting pp with", job_server.get_ncpus(), "workers"

    start_time = time.time()

    for i in range(50):
        run_time = time.time()
        jobs = [(child, job_server.submit(run_grn, 
                                          (child['genome'], 
                                           companies,
                                           delta),
                                           (),
                                           ("grn",)))
                                           for child in children]
        for child, result in jobs:
            child['fitness'], child['testfit'] = result()

        children = evo.iterate(children)
        print "gen", evo.gen_count, "time taken", str(time.time()-run_time), "total time", str(time.time()-start_time)

        if evo.adaptive:
            evo.adapt_mutation()

        res_file = open(filename,"a")
        print "best fitness:", evo.pop[-1]['fitness'], evo.pop[-1]['testfit']
        res_file.write(str(evo.pop[-1])+'\n')
        res_file.close()
            
        best_list.append(evo.pop[-1]['fitness'])
        mut_list.append(evo.mut_rate)

    mutfile = open('mutrate.txt','a')
    mutfile.write(str(mut_list)+'\n')
    mutfile.close()
    
if __name__ == "__main__":
    main()
