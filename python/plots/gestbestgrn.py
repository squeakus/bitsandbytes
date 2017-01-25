import os

#SYNCSIZE = ["/sync1","/sync10","/sync100","/sync500","/sync1000","/sync10000"]
SYNCSIZE = ["/sync1000"]
OFFSET = ['0','10','20','30','40','50','60']

def get_best(folder, outfilename):
    print outfilename
    os.chdir(folder)
    savelist = []
    outfile = open(outfilename,'w')
    
    for fileName in os.listdir('.'):
        if fileName.endswith('.dat'):
            savelist.append(fileName)

    for saved in savelist:
        tmpfitness = []
        results = open(saved,'r')
        results.readline()
        
        for line in results:
            if line.startswith('Rank'):
                best = line
        best = best.split('Genotype:')[1]
        best =  best.replace('[','').replace(']','')
        outfile.write(best+'\n')
    outfile.close()
        
for offset in OFFSET:
    print "getting best indivs for offset", offset
    EXP_LIST = []
    home_dir = os.getcwd()
    
    for syncsize in SYNCSIZE:
        EXP_LIST.append(syncsize+"offset"+offset)

    for experiment in EXP_LIST:
        get_best(home_dir + experiment, home_dir+'/'+offset+'.dat')
    os.chdir(home_dir)
