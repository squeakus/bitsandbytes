from deap import benchmarks, algorithms, base, creator, tools
import random, pylab

'''
Created on 13-11-2012
@author: wysek
'''

N = 80
Nbar = 40
GEN = 100
U = 0
V = 1

def my_rand():
    return random.random() * (V - U) - (V + U) / 2

def SPEA2():
    creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax) #@UndefinedVariable
    toolbox = base.Toolbox()
    toolbox.register("attr_float", my_rand)

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=20) #@UndefinedVariable
    toolbox.register("evaluate", benchmarks.zdt1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=0.5, low=U, up=V)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=U, up=V, indpb=1)
    toolbox.register("select", tools.selSPEA2)
    #toolbox.register("select", tools.selNSGA2)
    #binary tournament selection
    toolbox.register("selectTournament", tools.selTournament, tournsize=2)


    # Step 1 Initialization
    pop = toolbox.population(n=N)
    archive = []
    curr_gen = 1

    while True:
        # Step 2 Fitness assignement
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        for ind in archive:
            ind.fitness.values = toolbox.evaluate(ind)

        # Step 3 Environmental selection
        archive  = toolbox.select(pop + archive, k=Nbar)

        # Step 4 Termination
        if curr_gen >= GEN:
            final_set = archive
            break

        # Step 5 Mating Selection
        mating_pool = toolbox.selectTournament(archive, k=N)
        offspring_pool = map(toolbox.clone, mating_pool)

        # Step 6 Variation
        # crossover 100% and mutation 6%
        for child1, child2 in zip(offspring_pool[::2], offspring_pool[1::2]):
            toolbox.mate(child1, child2)

        for mutant in offspring_pool:
            if random.random() < 0.06:
                toolbox.mutate(mutant)

        pop = offspring_pool
        print "gen",curr_gen
        curr_gen += 1
#        x, y = [], []
#        for ind in pop:
#            x.append(ind.fitness.values[0])
#            y.append(ind.fitness.values[1])
#        pylab.scatter(x,y)
#        pylab.show()

    x, y = [], []
    for ind in final_set:
        x.append(ind.fitness.values[0])
        y.append(ind.fitness.values[1])
    return x, y

def NSGA2():
    creator.create("FitnessMax", base.Fitness, weights=(-1.0, -1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax) #@UndefinedVariable
    toolbox = base.Toolbox()
    toolbox.register("attr_float", my_rand)

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=20) #@UndefinedVariable
    toolbox.register("evaluate", benchmarks.zdt1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=0.5, low=U, up=V)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=U, up=V, indpb=1)
    #toolbox.register("select", tools.selSPEA2)
    toolbox.register("select", tools.selNSGA2)
    #binary tournament selection
    toolbox.register("selectTournament", tools.selTournament, tournsize=2)


    # Step 1 Initialization
    pop = toolbox.population(n=N)
    archive = []
    curr_gen = 1

    while True:
        # Step 2 Fitness assignement
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        for ind in archive:
            ind.fitness.values = toolbox.evaluate(ind)

        # Step 3 Environmental selection
        archive  = toolbox.select(pop + archive, k=Nbar)

        # Step 4 Termination
        if curr_gen >= GEN:
            final_set = archive
            break

        # Step 5 Mating Selection
        #offspring = tools.selTournamentDCD(pop, len(pop))
        #offspring = [toolbox.clone(ind) for ind in offspring]        
        mating_pool = toolbox.selectTournament(archive, k=N)
        offspring_pool = map(toolbox.clone, mating_pool)

        # Step 6 Variation
        # crossover 100% and mutation 6%
        for child1, child2 in zip(offspring_pool[::2], offspring_pool[1::2]):
            toolbox.mate(child1, child2)

        for mutant in offspring_pool:
            if random.random() < 0.06:
                toolbox.mutate(mutant)

        pop = offspring_pool
        print "gen",curr_gen
        curr_gen += 1
#        x, y = [], []
#        for ind in pop:
#            x.append(ind.fitness.values[0])
#            y.append(ind.fitness.values[1])
#        pylab.scatter(x,y)
#        pylab.show()

    x, y = [], []
    for ind in final_set:
        x.append(ind.fitness.values[0])
        y.append(ind.fitness.values[1])
    return x,y
        
if __name__ == '__main__':
    for i in range(5):
        x, y = NSGA2()
        pylab.scatter(x, y, color='red')
        x, y = SPEA2()
        pylab.scatter(x, y, color='blue')

    pylab.show()
