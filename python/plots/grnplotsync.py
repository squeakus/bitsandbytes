import graph, os

SYNCSIZE = ["/sync1","/sync10","/sync100","/sync500","/sync1000"]
OFFSET = ['0','100','200','300','400','500']

def graph_folder(folder):
    #print "collating folder:", folder
    os.chdir(folder)
    savelist =[]

    for fileName in os.listdir('.'):
        if fileName.endswith('.dat'):
            savelist.append(fileName)

    fitness = []
    for saved in savelist:
        tmpfitness = []
        results = open(saved,'r')
        results.readline()
        
        for line in results:
            result  = eval(line)
            tmpfitness.append(result['fitness'])
        fitness.append(tmpfitness)
    return fitness

def graph_offset(exp_list, offset):
    fitness = []
    home_dir = os.getcwd()

    for experiment in exp_list :
        result = graph_folder(home_dir+experiment)
        print experiment, len(result)
        fitness.append(result)

    legend = [x.replace('/','') for x in SYNCSIZE]

    os.chdir(home_dir)
    graph.multi_plot(fitness, "Offset "+offset, legend)

for offset in OFFSET:
    print "plotting offset", offset
    EXP_LIST = []
    for syncsize in SYNCSIZE:
        EXP_LIST.append(syncsize+"offset"+offset)
    graph_offset(EXP_LIST,offset)
