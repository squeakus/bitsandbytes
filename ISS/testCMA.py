import cma, random

def eval_soln(soln):
    if sum(soln) == 0:
        return random.random()
    else:
        return 1000

optim = cma.CMAEvolutionStrategy(10 * [0], 0.5, {'bounds': [0, 1]})
for i in range(10):
    print "generation", i
    solutions = optim.ask()
    fitnesses = []
    fitnessval = 1000
    for soln in solutions:
        
        #fitness = eval_soln(soln)
        fitness = fitnessval
        fitnessval -= 1
        #while fitness == 1000:
        #    soln = optim.ask(1)[0]
        #    fitness = eval_soln(soln)
        
        fitnesses.append(fitness)
        print "F",fitness
    optim.tell(solutions, fitnesses) 
print optim.result()[1]
