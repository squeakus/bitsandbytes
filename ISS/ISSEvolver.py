import random, re, evostrat, subprocess

def eval_config(genome, render=False):
    """
    Create a csv and then run simulator, extract power ratio
    First 3 digits rotate the ISS x,y,z axis
    """
    outfile = open('relaxed.csv', 'w')
    outfile.write("#header\n")
    configstr = "0,0,0" # start with locked configuration
    for codon in genome:
        configstr = configstr + ',' + str(codon)
    configstr += '\n'
    for i in range(92):
        outfile.write(configstr)
    outfile.close()

    if render:
        cmd = "java -jar ISSVis.jar -csv relaxed.csv -render"
    else:
        cmd = "java -jar ISSVis.jar -csv relaxed.csv"
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

def main():
    best_list, mut_list = [], []
    evo = evostrat.Evostrategy(10, 10)
    children = evo.iterate(evo.pop)

    for i in range(50):
        print "gen", i
        for child in children:
            child['fitness'] = eval_config(child['genome'])
        children = evo.iterate(children)


        if evo.adaptive:
            evo.adapt_mutation()

        best_list.append(evo.pop[-1]['fitness'])
        mut_list.append(evo.mut_rate)

    #save result
    #eval_config(evo.pop[-1]['genome'], True)
    print "best config:",evo.pop[-1]['fitness'], evo.pop[-1]['genome']
    bestgenome = evo.pop[-1]['genome']
    bestfit = evo.pop[-1]['fitness']
    print "best", bestfit, "genome", bestgenome
    beststr = str(bestfit) + " : " + str(bestgenome) + "\n"
    resfile = open("esresult.dat",'a')
    resfile.write(beststr)
    resfile.close()

    
if __name__ == "__main__":
#    random.seed(0)
    main()
