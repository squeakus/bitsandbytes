import graph, os

#SYNCSIZE = ["/sync1","/sync10","/sync100","/sync500","/sync1000"]
SYNCSIZE = ["/sync1000","/var1000"]
OFFSET = ['0','10','20','30','40','50','60']

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
            if not line.startswith('Rank'):
                result = float(line.split(' ')[0])
                tmpfitness.append(126.0 - result)
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
    #legend = ['1','10','100','500','1000']

    os.chdir(home_dir)
    graph.multi_plot(fitness, "Offset "+offset, legend)

for offset in OFFSET:
    print "plotting offset", offset
    EXP_LIST = []
    for syncsize in SYNCSIZE:
        EXP_LIST.append(syncsize+"offset"+offset)
    graph_offset(EXP_LIST,offset)
