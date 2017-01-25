import random, re, cma, subprocess, sys

def eval_config(name, beta, genome, render=False):
    """
    Create a csv and then run simulator, extract power ratio
    First 3 digits rotate the ISS x,y,z axis
    """
    name = name+str(beta)+".csv"
    outfile = open(name, 'w')
    outfile.write("#header\n")

    #generate line for each minute
    for minute in range(92):
        #alpha = (360.0/92.0) * minute #moving alpha
        alpha = 0
        configstr = str(alpha) + "," + str(beta) + ",0"

        for codon in genome:
            configstr = configstr + ',' + str(codon)
        configstr += '\n'
        outfile.write(configstr)
    outfile.close()
    cmd = "java -jar ISSVis.jar -csv "+ name
    if render:
        cmd = cmd + " -render"
        
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)

    # extract wattage from the result
    result = process.communicate()
    powerstr = result[0].split('\n')[2]
    powerstr = powerstr.replace(',','')
    fitness = re.findall(r'\d+.\d+', powerstr)[0]
    print "fitness", fitness
    return float(fitness)

def main(beta):
    optim = cma.CMAEvolutionStrategy(10 * [180], 45, {'bounds': [0, 360]})

    for i in range(100):
        print "generation", i
        solutions = optim.ask()
        fitnesses = []
        for soln in solutions:
            fitness = eval_config("test", beta, soln)
            minimize = 100000000 - fitness 
            fitnesses.append(minimize)
        optim.tell(solutions, fitnesses)

    #write results
    bestgenome = list(optim.result()[0])
    bestfit = 100000000 - optim.result()[1]
    print "best", bestfit, "genome", bestgenome
    beststr = str(bestfit) + " : " + str(bestgenome) + "\n"
    resfile = open("cmaresult"+str(beta)+".dat",'a')
    resfile.write(beststr)
    resfile.close()

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print "Usage: " + sys.argv[0] + " <beta>"
        sys.exit(1)
    beta = float(sys.argv[1])
    main(beta)

    
