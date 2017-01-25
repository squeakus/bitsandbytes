"""evolver: does the evolvy bit
Copyright (c) 2010 Jonathan Byrne, Erik Hemberg and James McDermott
Hereby licensed under the GNU GPL v3."""

from optparse import OptionParser
from nsga import *
from fitness import run_cmd
import sys, os, copy, random, math, datetime, time
import cessnafitness, bwbfitness, mig21fitness
import grammar as GRAMMAR

CODON_RANGE = 100 #randint needs a range
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_PROBABILITY = 0.015
CROSSOVER_PROBABILITY = 0.7
GRAMMAR_FILE = "grammars/bwbgrammar.bnf"

if len(sys.argv) < 2:
    print "Please set a random seed"
    exit()
else:
    random.seed(sys.argv[1])
    RUNID = sys.argv[1]
    PATHNAME = "results/run"+str(sys.argv[1])
    FITNESS_FUNCTION = bwbfitness.CFD_Fitness(runid=RUNID)

def main():
    cwd = os.getcwd()
    if cwd.startswith('/ichec/home'):
	print "Run it from the work dir!"
	exit()
    runstarttime = time.time()
        

    if os.path.exists(PATHNAME):
        print "path already exists! going to overwrite results"
        exit()
    else:
        os.makedirs(PATHNAME)
        
    BNF_GRAMMAR = GRAMMAR.Grammar(GRAMMAR_FILE)
    INDIVIDUALS = initialise_population(POPULATION_SIZE)
    LAST_POP = search_loop(GENERATIONS, INDIVIDUALS, BNF_GRAMMAR, tournament_selection, FITNESS_FUNCTION, PATHNAME)
    endtime = time.time() - runstarttime   
    print "total run took", endtime, "seconds"

class Individual(object):
    """A GE individual"""
    def __init__(self, genome, length=100):
        if genome == None:
            self.genome = [random.randint(0, CODON_RANGE)
                           for _ in xrange(length)]
        else:
            self.genome = copy.deepcopy(genome)

        self.dragmax = 5
        self.bad = default_fitness(FITNESS_FUNCTION.maximise)
        self.phenotype = None
        self.derivation_tree = None
        self.rank = None
        self.distance = None
        self.used_codons = 0
        self.fitness = [self.bad, self.bad]

    def __lt__(self, other):
        if FITNESS_FUNCTION.maximise:
            return self.fitness < other.fitness
        else:
            return other.fitness < self.fitness

    def __str__(self):
        return ("Individual: " + str(self.genome) + "; " + str(self.fitness))

    def set_values(self, values):
        self.phenotype = values['phenotype']
        self.used_codons = values['used_codons']
        self.derivation_tree = values['derivation_tree']
        
    def evaluate(self, fitness, idx):
        self.plane = fitness(self.phenotype, idx)
    
    def dominates(self, other):
        """ This is set to favour maximum fitness"""
        dominated = False
        won, lost = False, False
        for val in range(len(self.fitness)):
            if self.fitness[val] >= other.fitness[val]:
                won = True
            elif self.fitness[val] < other.fitness[val]:
                lost = True
        if won and not lost:
            dominated = True
        return dominated
    
def initialise_population(size=10):
    """Create a popultaion of size and return"""
    return [Individual(None) for _ in xrange(size)]

def ave(values):
    return float(sum(values)) / len(values)

def std(values, ave):
    return math.sqrt(float(sum((value - ave) ** 2
                     for value in values)) / len(values))
def print_stats(individuals, generation, pathname):
    best_lift = sorted(individuals, reverse=True,
                       key=lambda individuals: individuals.fitness[0])[0]
    best_drag = sorted(individuals, reverse=True,
                       key=lambda individuals: individuals.fitness[1])[0]
        
    ave_lift = ave([i.fitness[0] for i in individuals])
    std_lift = std([i.fitness[0] for i in individuals], ave_lift)
    ave_drag = ave([i.fitness[1] for i in individuals])
    std_drag = std([i.fitness[1] for i in individuals], ave_drag)
    
    out = "Gen:%d bestlift:%s bestdrag:%s avelift:%.1f+-%.1f avedrag:%.1f+-%.1f"% (generation, best_lift.fitness, best_drag.fitness, ave_lift, std_lift, ave_drag, std_drag)
    print out
    filename = pathname + "/results.dat"
    savefile = open(filename, 'a')
    savefile.write(out + '\n')
    savefile.close()
    
def default_fitness(maximise=False):
    if maximise:
        return - sys.maxint -1
    else:
        return sys.maxint

def int_flip_mutation(individual, use_prob=True):
    """Works per-codon, hence no need for "within_used" option."""
    if use_prob:
        for i in xrange(len(individual.genome)):
            if random.random() < MUTATION_PROBABILITY:
                individual.genome[i] = random.randint(0, CODON_RANGE)
    else:
        idx = random.randint(0, individual.used_codons - 1)
        individual.genome[idx] = individual.genome[idx] + 1
    return individual

def onepoint_crossover(p, q, within_used=True):
    """Given two individuals, create two children using one-point
    crossover and return them."""
    pc, qc = p.genome, q.genome

    if within_used:
        maxp, maxq = p.used_codons, q.used_codons
    else:
        maxp, maxq = len(pc), len(qc)

    # slice chromosome depending on probability
    point = random.randint(1, min(maxp,maxq)-1)
    if random.random() < CROSSOVER_PROBABILITY:
        c = pc[:point] + qc[point:]
        d = qc[:point] + pc[point:]
    else:
        c, d = pc, qc
    # Put the new chromosomes into new individuals
    return [Individual(c), Individual(d)]

def tournament_selection(population, tournament_size=3):
    """Given an entire population, draw <tournament_size> competitors
    randomly and return the best."""
    winners = []
    while len(winners) < POPULATION_SIZE:
        competitors = random.sample(population, tournament_size)
        competitors.sort(reverse=True)
        winners.append(competitors[0])
    return winners
    
def evaluate_fitness(individuals, grammar, fitness_function):
    """Perform the mapping and evaluate each individual"""
    for idx, ind in enumerate(individuals):
        generated_values = grammar.generate(ind.genome)
        ind.set_values(generated_values)
        ind.evaluate(fitness_function, idx)

    jobname = "planegen"+str(RUNID)+" "
    run_cmd('qsub -N '+jobname+' taskfarm.pbs')
    checkjobs(jobname)

    for idx, ind in enumerate(individuals):
        cfdid = "cfd%03d" % idx
        forces = get_forces(cfdid)
        ind.forces = forces
        ind.plane['lift'] = forces['lift']
        ind.plane['drag'] = forces['drag']
        ind.fitness = [ind.plane['lift'],ind.dragmax - ind.plane['drag']]

def checkjobs(jobname):
    print "checking for a job called", jobname
    jobsrunning = True
    starttime = time.time()

    while jobsrunning:
        qstat = run_cmd('qstat | grep '+jobname, qstat=True)
        qstat = qstat[0].split('\n')
        jobs = 0
        for line in qstat:
            if isinstance(line,str):
                if jobname in line:
                    jobs += 1
        time.sleep(30)
        if jobs == 0:
            jobsrunning = False
    endtime = time.time() - starttime
    print "generation took", endtime, "seconds"
    timefile = open('gentime.dat','a')
    timefile.write(str(endtime)+'\n')
    timefile.close()

def get_forces(cfdname):
    cfdfile = cfdname+'/postProcessing/forceCoeffs1/0/forceCoeffs.dat'
    if os.path.exists(cfdfile):
        """Parse the lift and drag from the postprocessing file"""
        forcefile = open(cfdfile)
        for line in forcefile:
            if not line.startswith('#'):
                line = line.strip()
                info = line.split('\t')
                info = [float(strval) for strval in info]
                forces = {'time':info[0], 
                          'pitchmoment':info[1],
                          'drag':info[2],
                          'lift':info[3], 
                          'frontlift':info[4], 
                          'rearlift':info[5]}
    else:
        print "Failed to write coefficient file"
        forces = {'time':0, 
                  'pitchmoment':0.0,
                  'drag':5.0,
                  'lift':0.0, 
                  'frontlift':0.0, 
                  'rearlift':0.0}        
    return forces

def search_loop(max_gens, individuals, grammar, selection, fit_func, pathname):
    """Loop over max generations"""
    #Evaluate initial population
    fronts = []
    evaluate_fitness(individuals, grammar, fit_func)
    best_ever = max(individuals)
    individuals.sort(reverse=True)
    print_stats(individuals,1, pathname)
    
    for generation in xrange(2, (max_gens + 1)):
        individuals, fronts, best_ever = step(
            individuals, fronts, grammar, selection, fit_func, best_ever)
        print_stats(individuals, generation, pathname)
        save_pop(individuals, generation, pathname)
    return individuals

def step(parent_pop, fronts, grammar, selection, fitness_function, best_ever):
    """perform single iteration and return next generation"""
    #Select parents, crossover and add to new pop
    pop_size = len(parent_pop)
    parents = selection(parent_pop)
    child_pop = []
    while len(child_pop) < POPULATION_SIZE:
        child_pop.extend(onepoint_crossover(*random.sample(parents, 2)))

    #Mutate the new population and evaluate
    child_pop = list(int_flip_mutation(child) for child in child_pop)
    evaluate_fitness(child_pop, grammar, fitness_function)

    #run fast non-dominated sort on child+parent pop
    total_pop = []
    fronts = []
    total_pop.extend(parent_pop)
    total_pop.extend(child_pop)
    fronts = fast_nondominated_sort(total_pop)
    #assign distance and append fronts to new population
    new_pop = []
    i = 0 #front counter
    while len(new_pop) + len(fronts[i]) <= pop_size:
        crowding_distance_assignment(fronts[i])
        new_pop.extend(fronts[i])
        i += 1
    #filling up pop with the final front
    crowding_distance_assignment(fronts[i])
    new_pop.extend(fronts[i][0: pop_size - len(new_pop)])
    best_ever = max(best_ever, max(parents))
    return new_pop, fronts, best_ever

def save_pop(population, gen, pathname):
    filename = pathname + "/gen%03d.dat" % gen
    savefile = open(filename, 'w')
    
    for indiv in population:
        indiv.plane['rank'] = indiv.rank
        indiv.plane['genome'] = indiv.genome
        indiv.plane['fitness'] = indiv.fitness
        savefile.write(str(indiv.plane) + '\n')
    savefile.close()

if __name__ == "__main__":
    main()
