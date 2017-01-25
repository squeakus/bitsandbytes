#!/usr/bin/python
import pylab, os, sys
import grammar as GRAMMAR
from evolver import *

LIFT, DRAG = 4.21913, 4.488744 #cessnalift
#LIFT, DRAG = 2.40721, 4.783129 #bwblift
#LIFT, DRAG = 0.00201827, 4.9683871 #miglift
#LIFT, DRAG = 2.3643601, 4.508076 #bwb castellated


def main():
    if len(sys.argv) < 3:
        print "please specify function and results folder"
        exit()

    func_name = sys.argv[1]
    result_name = sys.argv[2]
    globals()[func_name](result_name)

def plotrun(name):
    resultsfolder = "results/"+name+"/"
    resultsfiles = []
    
    for filename in os.listdir(resultsfolder):
        if filename.startswith('gen') and filename.endswith('.dat'):
            resultsfiles.append(resultsfolder+filename)

    lift_range, drag_range = get_ranges(resultsfiles)
    print "lift/drag range", lift_range, drag_range

    for filename in resultsfiles:
        scatterplot(filename,'red', lift_range, drag_range)
        pareto_frontier(filename, lift_range, drag_range, 'red')
        pylab.xlabel("Lift Maximisation",fontsize=22)
        pylab.ylabel("Drag Minimisation",fontsize=22)

        pylab.savefig(filename.rstrip('.dat') + '.png')
        pylab.clf()

def finalgen(names):
    names = eval(names)
    totaleff = []
    for name in names:
        resultsfolder = "results/"+name+"/"
        final = resultsfolder + "gen049.dat"
        population = []
        name = final.rstrip('.dat')
        efflist = []

        resultsfile = open(final, 'r')
        for line in resultsfile:
            population.append(eval(line))

        for indiv in population:
            # if "fullrandom" in name:
            #     print "found", name
            #     lift = indiv['fitness'][0] - 0.5
            # else:
            lift = indiv['fitness'][0]
            drag = 5.0 - indiv['fitness'][1]
            efficiency = lift/drag
            efflist.append(efficiency)
        aveeff = ave(efflist)
        stdeff = std(efflist,aveeff)
        print "efficieny average", aveeff, "+-", stdeff
        totaleff.append(efflist)
    pylab.boxplot(totaleff)
    pylab.show()
    
    bwblift, bwbdrag = LIFT, 5 - DRAG
    print "bwbefficiency", bwblift/bwbdrag

def parse_pop(filename):
    population = []
    resultsfile = open(filename, 'r')
    for line in resultsfile:
        population.append(eval(line))
    return population

    
def graphruns(names):
    names = eval(names)
    liftaverages, dragaverages = [],[]
    liftdevs, dragdevs = [],[]
    for name in names:
        resultsfolder = "results/"+name+"/"
        files = sorted(os.listdir(resultsfolder))
        liftave, dragave = [], []
        liftstd, dragstd = [], []
        
        for filename in files:
            print "filename:", filename
            if filename.startswith('gen') and filename.endswith('.dat'):
                population = parse_pop(resultsfolder+filename)
                lift = [ind['fitness'][0] for ind in population]
                drag = [ind['fitness'][1] for ind in population]
                liftave.append(ave(lift))
                dragave.append(ave(drag))
                liftstd.append(std(lift,ave(lift)))
                dragstd.append(std(drag,ave(drag)))
                
        liftaverages.append(liftave)
        dragaverages.append(dragave)
        liftdevs.append(liftstd)
        dragdevs.append(dragstd)

    names = ["Evolved", "Random"]
    ltype = ['-','--']
    colors = ['red','black']
    xval = range(0,49)[::5]
    for idx,x in enumerate(liftaverages):
        pylab.plot(x,color=colors[idx],ls=ltype[idx],linewidth=2.0)
        #pylab.title("Average Lift Maximisation of Population")
        pylab.xlabel("Generations",fontsize=20)
        pylab.ylabel("Lift Maximisation",fontsize=20)

    pylab.legend(names,loc='best')

    #add the errorbars after the legend
    for idx,x in enumerate(liftaverages):
        x = x[::5]
        dev = liftdevs[idx][::5]
        for cnt in range(len(x)):
                pylab.errorbar(xval[cnt],x[cnt], 
                               yerr=dev[cnt], color=colors[idx])
    pylab.savefig("lift.png")
    pylab.clf()
    
    for idx,x in enumerate(dragaverages):
        pylab.plot(x,color=colors[idx],ls=ltype[idx],linewidth=2.0)
        #pylab.title("Average Drag Minimisation of Population")
        pylab.xlabel("Generations",fontsize=20)
        pylab.ylabel("Drag Minimisation",fontsize=20)

    pylab.legend(names,loc=3)
    #add the errorbars after the legend
    for idx,x in enumerate(dragaverages):
        x = x[::5]
        dev = dragdevs[idx][::5]
        for cnt in range(len(x)):
                pylab.errorbar(xval[cnt],x[cnt], yerr=dev[cnt],
                               color=colors[idx])
    pylab.savefig("drag.png")
    #pylab.show()

    #pylab.plot(dragaverages)
            # pylab.xlabel("Lift Maximisation",fontsize=22)
            # pylab.ylabel("Drag Minimisation",fontsize=22)

            # # showing the original design fitness
            # origlift = normalize(LIFT, lift_range)
            # origdrag = normalize(DRAG, drag_range)
            # #print "original lift/drag", origlift, origdrag
            # handle = pylab.scatter(origlift, origdrag, 
            #                        s=100, c="blue", label="original")
            # handles.append(handle)
            # #add the legend
            # #legendnames = ["Evolved", "Random", "Original"]
            # legendnames = names
            # pylab.legend(handles,legendnames, scatterpoints=1, loc='best')
            # pylab.savefig(filename.rstrip('.dat') + '.png')
            # pylab.clf()


    
def plotruns(names):
    names = eval(names)
    resultsfiles,colors, handles = [],[],[]
    for name in names:
        resultsfolder = "results/"+name+"/"
        for filename in os.listdir(resultsfolder):
            if filename.startswith('gen') and filename.endswith('.dat'):
                resultsfiles.append(resultsfolder+filename)

    lift_range, drag_range = get_ranges(resultsfiles)
    print "lift/drag range", lift_range, drag_range
    
    resultsfolder = "results/"+names[0]+"/"
    resultsfiles = []
    for filename in os.listdir(resultsfolder):
        if filename.startswith('gen') and filename.endswith('.dat'):
            for idx, name in enumerate(names):
                resultsfile ="results/"+name+"/"+filename
                if os.path.exists(resultsfile):
                    color = pylab.cm.gist_rainbow(1.*idx/len(names))
                    handle = scatterplot(resultsfile, color, lift_range, 
                                         drag_range, name)
                    pareto_frontier(resultsfile, lift_range, drag_range, color)
                    if not color in colors:
                        colors.append(color)
                        handles.append(handle)
                        
            pylab.xlabel("Lift Maximisation",fontsize=22)
            pylab.ylabel("Drag Minimisation",fontsize=22)

            # showing the original design fitness
            origlift = normalize(LIFT, lift_range)
            origdrag = normalize(DRAG, drag_range)
            #print "original lift/drag", origlift, origdrag
            handle = pylab.scatter(origlift, origdrag, 
                                   s=100, c="black", label="original")
            handles.append(handle)
            #add the legend

            legendnames = ["Evolved Cessna", "Random Cessna", "Original"]
            #legendnames = names
            pylab.legend(handles,legendnames, scatterpoints=1, loc='best')
            pylab.savefig(filename.rstrip('.dat') + '.png')
            pylab.clf()

def pareto_frontier(filename, lift_range, drag_range, color, maxX = True, maxY = True):
    population = []
    Xs, Ys = [],[]
    name = filename.rstrip('.dat')
    resultsfile = open(filename, 'r')
    for line in resultsfile:
        population.append(eval(line))

    for indiv in population:
        if not lift_range == None:
            lift = normalize(indiv['fitness'][0], lift_range)
            drag = normalize(indiv['fitness'][1], drag_range)
        else: 
            lift = indiv['fitness'][0]
            drag = indiv['fitness'][1]
        Xs.append(lift)
        Ys.append(drag)
        
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of
                p_front.append(pair) #and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of 
                p_front.append(pair) # and add them to the Pareto frontier
# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    pylab.plot(p_frontX, p_frontY, color=color)

            
def normalize(value, old_range):
    #reduce from range to 0,1
    if value < old_range[0]: value = old_range[0]
    if value > old_range[1]: value = old_range[1]
    
    normalized = (float(value) -old_range[0]) / (old_range[1] - old_range[0])
    return normalized
    
def scatterplot(filename, color, lift_range=None, drag_range=None, run=None):
    
    population = []
    name = filename.rstrip('.dat')
    resultsfile = open(filename, 'r')
    for line in resultsfile:
        population.append(eval(line))

    for indiv in population:
        if not lift_range == None:
            lift = normalize(indiv['fitness'][0], lift_range)
            drag = normalize(indiv['fitness'][1], drag_range)
        else:
            lift = indiv['fitness'][0]
            drag = indiv['fitness'][1]
        
        x = pylab.scatter(lift, drag, s=100, c=color, alpha=0.7, label=run)


    pylab.grid(True)
    #pylab.title("Pareto Optimisation for "+name)
    
    if not lift_range == None:
        pylab.xlim(0,1.1)
        pylab.ylim(0,1.1)
    return x

    
def get_ranges(resultsfiles):
    population = []
    liftfit = []
    dragfit = []
    
    for filename in resultsfiles:
        resultsfile = open(filename, 'r')
        for line in resultsfile:
            population.append(eval(line))

        for indiv in population:
            liftfit.append(indiv['fitness'][0])
            dragfit.append(indiv['fitness'][1])
    lift_range = (min(liftfit),max(liftfit))
    drag_range = (min(dragfit),max(dragfit))
    return lift_range, drag_range

def lastgen(resultsfolder):
    PATHNAME = "results/"+resultsfolder
    print "pathname", PATHNAME
    filename = get_last_gen(PATHNAME)
    lastgen = int(filename.rstrip('.dat').lstrip('gen'))
    print "the last generation was:", lastgen
    parsed_pop = parse_pop(PATHNAME+'/'+filename)
    sorted_pop = sort_pop(parsed_pop)
    
    BNF_GRAMMAR = GRAMMAR.Grammar(GRAMMAR_FILE)
    INDIVIDUALS = reinitialise_pop(sorted_pop)
    FITNESS_FUNCTION = bwbfitness.CFD_Fitness(debug=True,foampng=False)
    evaluate_fitness(INDIVIDUALS, BNF_GRAMMAR, FITNESS_FUNCTION)

def parse_pop(filename):
    population = []
    name = filename.rstrip('.dat')
    resultsfile = open(filename, 'r')
    for line in resultsfile:
        population.append(eval(line))
    return population

def sort_pop(population):
    sorted_pop = sorted(population, key=lambda item: item['fitness'][0])
    return sorted_pop
    
def reinitialise_pop(parsed_pop):
    newpop = []
    for indiv in parsed_pop:
        newpop.append(Individual(indiv['genome']))
    return newpop

def get_last_gen(pathname):
    files = os.listdir(pathname)
    datfiles =  [i for i in files if i.startswith('gen') and i.endswith('.dat')]
    datfiles = sorted(datfiles)
    print "starting at", datfiles[-1], "in", pathname
    return datfiles[-1]
    
if __name__ == '__main__':
    main()
