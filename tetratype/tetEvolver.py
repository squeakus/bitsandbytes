#! /usr/bin/env python
# evolver: does the evolvy bit
# Copyright (c) 2010 Jonathan Byrne, Erik Hemberg and James McDermott
# Hereby licensed under the GNU GPL v3.

import sys, os, copy, random, math, time, tetragraph, analyser
import multiprocessing

#import cProfile


class Individual(object):
    """A GE individual"""

    def __init__(self, genome, length):
        if genome == None:
            self.genome = [random.randint(0, CODON_SIZE)
                           for _ in xrange(length)]
        else:
            self.genome = copy.deepcopy(genome)
        self.phenotype = tetragraph.Tetragraph()
        self.distance = None
        self.bad = default_fitness(CalculateFitness.maximise)
        self.used_codons = 0
        self.fitness = self.bad

    def save_result(self, result):
        """assign values from result array to individual"""
        self.fitness = result

    def __lt__(self, other):
        if CalculateFitness.maximise:
            return self.fitness < other.fitness
        else:
            return other.fitness < self.fitness

    def __str__(self):
        return ("Individual: " + str(self.genome) + "; "
                + str(self.fitness))


def graph_distance(graph_a, graph_b):
    """calculates fitness value by recording distance between nodes
    on the largest graph with the nearest node on the smaller graph"""
    if graph_a.size() > graph_b.size():
        bigger, smaller = graph_a, graph_b
    else:
        bigger, smaller = graph_b, graph_a

    total_distance = 0
    for node_id in bigger.node:
        pos = bigger.node[node_id]['xyz']
        distance = smaller.nearest_node(pos[0], pos[1], pos[2])
        total_distance  += distance
    total_distance = round(total_distance, 0)
    return total_distance


def structural_fitness(graph):
    azr = analyser.Analyser('test',"moo",True)
    azr.myGraph=graph
    azr.parse_graph(graph)
    azr.apply_stresses()
    azr.create_slf_file()
    azr.test_slf_file()
    azr.parse_results()
    fitness, weight = azr.calculate_fitness()
    return fitness

class CalculateFitness():
    """Fitness function for testing generated mesh programs. """
    maximise = False # false = smaller is better
    def __init__(self, phenotype, target):
        self.test_graph = phenotype
        self.target = target

    def calculate(self):
        """will be used to group multiple fitness values"""
        #fitness = graph_distance(self.test_graph, self.target)
        print "calculating fitness"
        fitness = structural_fitness(self.test_graph)
        return fitness


########And now for the evolution###########
def evaluate(indiv, phenotype, target):
    """threaded evaluation method"""
    indiv.phenotype = phenotype.copy()
    indiv.phenotype = indiv.phenotype.generate_graph(indiv.genome)
    calc_fit = CalculateFitness(indiv.phenotype, target)
    indiv.fitness = calc_fit.calculate()
    return indiv.fitness


def evaluate_fitness(individuals, phenotype, target):
    """uses all available cores to calculate fitness of a population"""
    if MULTI_CORE:
        #use all available cores
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        # Perform the mapping for each individual
        for indiv in individuals:
            pool.apply_async(evaluate, (indiv, phenotype, target,), 
                             callback=indiv.save_result)
        pool.close()
        pool.join()
    else:
        for indiv in individuals:
            print "evaluating:", len(individuals)
            fit_val = evaluate(indiv, phenotype, target)
            indiv.fitness = fit_val


def initialise_population(size, length):
    """Create a popultaion of size and return"""
    return [Individual(None, length) for _ in xrange(size)]


def generational_replacement(new_pop, individuals):
    """only keeps elites from parent population"""
    individuals.sort(reverse=True)
    for ind in individuals[:ELITE_SIZE]:
        new_pop.append(copy.copy(ind))
    new_pop.sort(reverse=True)
    return new_pop[:GENERATION_SIZE]

def steady_state_replacement(new_pop, individuals):
    """takes best of both parent and child populations"""
    individuals.sort(reverse=True)
    individuals[-1] = max(new_pop + individuals[-1:])
    return individuals

def default_fitness(maximise=False):
    "used for maximising or minimising fitness, depending on problem"
    if maximise:
        return -DEFAULT_FIT
    else:
        return DEFAULT_FIT

def int_flip_mutation(individual, use_prob = True):
    """Mutate the individual by randomly chosing a new int with
    probability p_mut. Works per-codon, hence no need for
    "within_used" option."""
    if use_prob:
        for i in xrange(len(individual.genome)):
            if random.random() < MUTATION_PROBABILITY:
                individual.genome[i] = random.randint(0, CODON_SIZE)
    else:
        idx = random.randint(0, individual.used_codons-1)
        individual.genome[idx] = individual.genome[idx]+1
    return individual

# Two selection methods: tournament and truncation
def tournament_selection(population, tournament_size=3):
    """Given an entire population, draw <tournament_size> competitors
    randomly and return the best."""
    winners = []
    while len(winners) < GENERATION_SIZE:
        competitors = random.sample(population, tournament_size)
        competitors.sort(reverse=True)
        winners.append(competitors[0])
    return winners

def truncation_selection(population, proportion=0.5):
    """Given an entire population, return the best <proportion> of
    them."""
    population.sort(reverse=True)
    cutoff = int(len(population) * float(proportion))
    return population[0:cutoff]

def onepoint_crossover(p, q):
    """Given two individuals, create two children using one-point
    crossover and return them."""
    # Get the chromosomes
    pc, qc = p.genome, q.genome
    # Uniformly generate crossover points. If within_used==True,
    # points will be within the used section.
    if not len(pc) == len(qc):
        print "lengths not equal!"
    maxp = len(pc)

    pt = random.randint(1, maxp)
    # Make new chromosomes by crossover: these slices perform copies
    if random.random() < CROSSOVER_PROBABILITY:
        c = pc[:pt] + qc[pt:]
        d = qc[:pt] + pc[pt:]
    else:
        c, d = pc, qc
    # Put the new chromosomes into new individuals
    return [Individual(c, len(c)), Individual(d, len(d))]


def print_stats(generation, individuals, start_time, phenotype):
    """writes stats to screen and to a file"""
    def ave(values):
        return float(sum(values))/len(values)
    def std(values, ave):
        return math.sqrt(float(sum((value-ave)**2 for value in values))
                         /len(values))

    ave_fit = ave([i.fitness for i in individuals if i.fitness is not None])
    std_fit = std([i.fitness for i in individuals if i.fitness is not None], 
                  ave_fit)
    time_taken = time.time()-start_time
    print("Gen:%d evals:%d ave:%.2f+-%.2f best: %s time: %.2f" % (generation, (GENERATION_SIZE*generation), ave_fit, std_fit, individuals[0].fitness, time_taken))
    individuals[0].phenotype = phenotype.copy()
    individuals[0].phenotype = individuals[0].phenotype.generate_graph(individuals[0].genome)
    individuals[0].phenotype.save_graph(generation)
    if not LOG_FILE == None:
        result = str(generation)+","+str(int(individuals[0].fitness))+","+str(int(ave_fit))+","+str(int(std_fit))+"\n"  
        if generation == 1:
            stats_file = open(LOG_FILE, 'w')
            stats_file.write(result)
            stats_file.close()
        else:            
            stats_file = open(LOG_FILE, 'a')
            stats_file.write(result)
            stats_file.close()

def step(individuals, phenotype, target, replacement, selection, best_ever):
    """performs a single iteration of the evolutionary algorithm"""
    #Select parents
    parents = selection(individuals)
    #Cross over parents and add to the new population
    new_pop = []
    while len(new_pop) < GENERATION_SIZE:
        new_pop.extend(onepoint_crossover(*random.sample(parents, 2)))
    #Mutate the new population
    new_pop = list(map(int_flip_mutation, new_pop))
    #new_pop = list(int_flip_mutation(x) for x in new_pop)  
    #Evaluate the fitness of the new population
    evaluate_fitness(new_pop, phenotype, target)
    #Replace the sorted individuals with the new populations
    individuals = replacement(new_pop, individuals)
    best_ever = max(best_ever, max(individuals))
    print "step"
    return individuals, best_ever

def search_loop(max_generations, individuals, phenotype, target, replacement, selection):
    """Loop over max generations"""
    #Evaluate initial population
    start_time = time.time()
    evaluate_fitness(individuals, phenotype, target)
    best_ever = max(individuals)
    individuals.sort(reverse=True)
    print_stats(1, individuals, start_time, phenotype)
    for generation in xrange(2, (max_generations+1)):
        start_time = time.time()
        print "gen:", generation
        individuals, best_ever = step(
            individuals, phenotype, target, replacement, selection, best_ever)
        print_stats(generation, individuals, start_time, phenotype)
    return best_ever

#GE Properties
SAVE_BEST = True
CODON_SIZE = 1
ELITE_SIZE = 10
POPULATION_SIZE = 10
GENERATION_SIZE = 10
GENERATIONS = 100
LOG_FILE = None 
MULTI_CORE = False
DEFAULT_FIT = 100000
MUTATION_PROBABILITY = 0.015
CROSSOVER_PROBABILITY = 0.7
IMG_COUNTER = 0

# Run program
def mane():
    # create phenoptype and target graph
    depth = 3
    length = 2000 #make it big to keep it in integers
    height = int((math.sqrt(2)*length)/2)
    gridOrig = [0, 0, height]
    phenotype = tetragraph.Tetragraph()
    phenotype.grid(phenotype.octahedron, gridOrig, length, 10)
    target = tetragraph.Tetragraph()
    #easy target
    #target.grid(target.octahedron,gridOrig,length,5)
    #hard target, a recursive regular tetrahedron
    #target.recurse(target.regular,[0,0,0],length,depth)
    #target.save_graph(0000)
    max_codons = phenotype.number_of_nodes()
    individuals = initialise_population(POPULATION_SIZE, max_codons)
    
    if MULTI_CORE:
        print "Core count:", str(multiprocessing.cpu_count())
    search_loop(GENERATIONS,individuals, phenotype, target, generational_replacement, tournament_selection)
  
if __name__ == "__main__":
    import getopt
    try:
        OPTS, ARGS = getopt.getopt(sys.argv[1:], "p:g:e:m:x:b:l:f:", ["population", "generations", "elite_size", "mutation", "crossover", "bnf_grammar", "log_file" , "front_folder"])
    except getopt.GetoptError, err:
        print(str(err))
        sys.exit(2)
    print "options:",OPTS
    for o, a in OPTS:
        if o in ("-p", "--population"):
            POPULATION_SIZE = int(a)
            GENERATION_SIZE = int(a)
        elif o in ("-g", "--generations"):
            GENERATIONS = int(a)
        elif o in ("-e", "--elite_size"):
            ELITE_SIZE = int(a)
        elif o in ("-m", "--mutation"):
            MUTATION_PROBABILITY = float(a)
        elif o in ("-x", "--crossover"):
            CROSSOVER_PROBABILITY = float(a)
        elif o in ("-b", "--bnf_grammar"):
            GRAMMAR_FILE = a
        elif o in ("-l", "--log_file"):
            LOG_FILE = a
            divider = LOG_FILE.find('/')
            if not divider == -1:
                folder = LOG_FILE[:divider]
                print "found a folder", folder
                if not os.path.isdir(folder):
                    print "folder doesn't exist, creating folder"
                    os.makedirs(folder)

        elif o in ("-f", "--front_folder"):
            FRONT_FOLDER = os.getcwd()+'/'+a
            print "writing fronts in folder:", FRONT_FOLDER
            if not os.path.isdir(FRONT_FOLDER):
                print "folder doesn't exist, creating folder"
                os.makedirs(FRONT_FOLDER)
        else:
            assert False, "unhandeled option"
    #cProfile.run('mane()')
    mane()
