"""evolver: does the evolvy bit
Copyright (c) 2010 Jonathan Byrne, Erik Hemberg and James McDermott
Hereby licensed under the GNU GPL v3."""

import sys, os, copy, random, math, datetime
import optimizer as OPT
import analyser as AZR
#random.seed(1)


class StructuralFitness():
    """Fitness function for testing generated mesh programs. """

    def __init__(self):
        self.maximise = False # false = smaller is better

    def __call__(self, unique_id, program):
        analyser = AZR.Analyser(unique_id, program)
        fitness_a, fitness_b, fitness_c = analyser.test_mesh()
        return fitness_a, fitness_b, fitness_c


class Individual(object):
    """A GE individual"""

    def __init__(self, genome, length=200):
        if genome == None:
            self.genome = [random.randint(0, CODON_SIZE)
                           for _ in xrange(length)]
        else:
            self.genome = copy.deepcopy(genome)
        self.bad = default_fitness(FITNESS_FUNCTION.maximise)
        self.uid = None
        self.phenotype = None
        self.derivation_tree = None
        self.rank = None
        self.distance = None
        self.fitness_a = self.bad
        self.fitness_b = self.bad
        self.fitness_c = self.bad
        self.used_codons = 0
        self.nodal_codons = []
        self.struct_codons = []
        self.condon_list = []

        self.fitness = [int(self.fitness_a), int(self.fitness_b), int(self.fitness_c)]

    def __lt__(self, other):
        if FITNESS_FUNCTION.maximise:
            return self.fitness < other.fitness
        else:
            return other.fitness < self.fitness

    def __str__(self):
        return ("Individual: " + " uid: " + str(self.uid)
                + str(self.genome) + "; " + str(self.fitness))

    def evaluate(self, fitness):
        #self.fitness_a, self.fitness_b, self.fitness_c = fitness(self.uid, self.phenotype)
        self.fitness_a, self.fitness_b, self.fitness_c = 10000,10000,10000
        self.fitness = [int(self.fitness_a), int(self.fitness_b), int(self.fitness_c)]

    def set_values(self, values):
        self.phenotype = values['phenotype']
        self.used_codons = values['used_codons']
        self.nodal_codons = values['nodal_codons']
        self.struct_codons = values['struct_codons']
        self.derivation_tree = values['derivation_tree']

    def dominates(self, other):
        dominated = False
        won, lost = False, False
        for val in range(len(self.fitness)):
            if self.fitness[val] <= other.fitness[val]:
                won = True
            elif self.fitness[val] > other.fitness[val]:
                lost = True
        if won and not lost:
            dominated = True
        return dominated


def initialise_population(size=10):
    """Create a popultaion of size and return"""
    return [Individual(None) for _ in xrange(size)]


def print_stats(generation, individuals):
    def ave(values):
        return float(sum(values)) / len(values)

    def std(values, ave):
        return math.sqrt(float(sum((value - ave) ** 2
                                   for value in values)) / len(values))
    ave_fitness_b = ave([i.fitness_b for i in individuals
                      if i.phenotype is not None])
    std_fitness_b = std([i.fitness_b for i in individuals
                      if i.phenotype is not None], ave_fitness_b)
    ave_fit = ave([i.fitness[1] for i in individuals
                   if i.phenotype is not None])
    std_fit = std([i.fitness[1] for i in individuals
                   if i.phenotype is not None], ave_fit)
    ave_used_codons = ave([i.used_codons for i in individuals
                           if i.phenotype is not None])
    std_used_codons = std([i.used_codons for i in individuals
                           if i.phenotype is not None], ave_used_codons)
    print("Gen:%d best compliance, deflection (mm) and weight (kg):%s Avg. deflection (mm):%.1f+-%.1f"
          % (generation, individuals[0].fitness, ave_fit, std_fit))

    if SAVE_BEST:
        print "saving best individual"
        best_mesh = AZR.Analyser(0,individuals[0].phenotype)
        filename = './saved/best.' + str(generation)
        best_mesh.create_mesh(filename)


def default_fitness(maximise=False):
    if maximise:
        return - DEFAULT_FIT
    else:
        return DEFAULT_FIT


def fast_nondominated_sort(pop):
    """takes a population and uses nsga2 algorithm to create a list of
    fronts"""
    fronts = []
    dom_list = dict()
    for A in pop:
        dom_list[hash(A)] = []
        A.domCount = 0
        for B in pop:
            if A.dominates(B):
                dom_list[hash(A)].append(B)
            elif B.dominates(A):
                A.domCount += 1
        if A.domCount == 0:
            A.rank = 1
            fronts.append([]) # add new front
            fronts[0].append(A) # add to first front

    i = 0 #front counter
    while len(fronts[i]) != 0:
        new_pop = []
        for A in fronts[i]:
            for B in dom_list[hash(A)]:
                B.domCount -= 1
                if B.domCount == 0:
                    B.rank = i + 1
                    new_pop.append(B)
        i += 1
        fronts.append([])
        fronts[i] = new_pop
    return fronts


def crowded_comparasion_operator(x, y):
    if x.rank < y.rank:
        return 1
    elif (x.rank == y.rank):
        if x.distance > y.distance:
            return -1
        else:
            return 1


def crowding_distance_assignment(pop):
    """calulates distances between individuals"""
    pop_size = len(pop)
    for indiv in pop:
        indiv.distance = 0
    #assign cumulative distance to each individual
    for val in range(len(pop[0].fitness)):
        pop.sort(lambda x, y: cmp(x.fitness[val], y.fitness[val]))
        #always include boundary points
        pop[0].distance = 9999
        pop[pop_size-1].distance = 9999
        for i in range(2, pop_size-1):
            pop[i].distance += (pop[i + 1].fitness[val]
                                - pop[i - 1].fitness[val])
        #send back a sorted pop
    pop.sort(crowded_comparasion_operator)


def count_fronts(fronts):
    front_cnt = 0
    for front in fronts:
        if len(front) > 0:
            front_cnt += 1
    print str(front_cnt), "valid fronts out of", len(fronts)
    return front_cnt


def write_front(idx, fronts):
    filename = FRONT_FOLDER + "/Gen" + repr(idx).zfill(3) + ".dat"
    front_file = open(filename, "w")
    for front in fronts:
        for indiv in front:
            t = (repr(indiv.fitness[0]).rjust(5) + " "
                 + repr(indiv.fitness[1]).rjust(5) + " "
                 + repr(indiv.fitness[2]).rjust(5) + "\n")
            front_file.write(t)
    front_file.close()


def write_fronts(fronts):
    for idx, front in enumerate(fronts):
        filename = FRONT_FOLDER + "/Front" + repr(idx).zfill(3) + ".dat"
        front_file = open(filename, "w")
        for indiv in front:
            t = (repr(indiv.fitness[0]).rjust(5) + " "
                 + repr(indiv.fitness[1]).rjust(5) + " "
                 + repr(indiv.fitness[2]).rjust(5) + "\n")
            front_file.write(t)
        front_file.close()

def graph_fronts(fronts):
    p = subprocess.Popen(" ".join(current_cmd), stdin=subprocess.PIPE, stdout=_file, shell=True)
    gema_master = {'cmd':['/Applications/MATLAB74/bin/matlab', '-nosplash', '-nodisplay', '-nojvm', '-logfile', os.path.join(OUT_DIR,'/master.log'), '-r', '\"gema_multicore_wrapper(\''+os.path.join(GEMA_DIR, 'gema_parameters.dat')+'\',\''+OUT_DIR+'\')\"'],
            }

def write_pop(idx, pop):
    filename = FRONT_FOLDER + "/Pop" + repr(idx).zfill(3) + ".dat"
    pop_file = open(filename, "w")
    for indiv in pop:
        t = (repr(indiv.fitness[0]).rjust(5) + " "
             + repr(indiv.fitness[1]).rjust(5) + " "
             + repr(indiv.fitness[2]).rjust(5) + "\n")
        pop_file.write(t)
    pop_file.close()


def write_mesh(fronts, name):
    counter = 0
    for front in fronts:
        for indiv in front:
            mesh = AZR.Analyser(indiv.uid, str(indiv.phenotype))
            mesh.create_graph()
            filename = FRONT_FOLDER + "/" + name + "." + str(counter)
            mesh.create_mesh(filename)
            counter += 1


def int_flip_mutation(individual, use_prob=True):
    """Mutate the individual by randomly chosing a new int with
    probability p_mut. Works per-codon, hence no need for
    "within_used" option."""
    if use_prob:
        for i in xrange(len(individual.genome)):
            if random.random() < MUTATION_PROBABILITY:
                individual.genome[i] = random.randint(0, CODON_SIZE)
    else:
        idx = random.randint(0, individual.used_codons - 1)
        individual.genome[idx] = individual.genome[idx] + 1
    return individual


def nodal_mutation(individual, use_prob=True):
    """only mutates codons encoding for nodal rules"""
    print "in nodal"
    if use_prob:
        print "using prob"
        for _ in xrange(len(individual.genome)):
            print "nodal_length", len(individual.nodal_codons)
            idx = random.choice(individual.nodal_codons)
            if random.random() < MUTATION_PROBABILITY:
                individual.genome[idx] = random.randint(0, CODON_SIZE)
    else:
        idx = random.choice(individual.nodal_codons)
        print "mutating node codon", idx
        individual.genome[idx] = individual.genome[idx] + 1
    return individual


def struct_mutation(individual, use_prob=True):
    """only mutates codons encoding for structural rules"""
    if use_prob:
        for _ in xrange(len(individual.genome)):
            idx = random.choice(individual.struct_codons)
            if random.random() < MUTATION_PROBABILITY:
                individual.genome[idx] = random.randint(0, CODON_SIZE)
    else:
        idx = random.choice(individual.struct_codons)
        print "mutating struct codon", idx
        individual.genome[idx] = individual.genome[idx] + 1
    return individual


def mutate_individual(ind, grammar, mut_op):
    """checks mutation operator and calls correct function """
#    before = ind.derivation_tree.textual_tree_view().splitlines()
    if mut_op == "int":
        ind = int_flip_mutation(ind, False)
    elif mut_op == "nodal":
        ind = nodal_mutation(ind, False)
    elif mut_op == "struct":
        ind = struct_mutation(ind, False)
    generated_values = grammar.generate(ind.genome)
    ind.set_values(generated_values)
#    after = ind.derivation_tree.textual_tree_view().splitlines()
#    d = difflib.Differ()
#    differences  = list(d.compare(before,after))
#    for line in differences:
#        if line.startswith('-')or line.startswith('+'):
#            print line
    analyser = AZR.Analyser(ind.uid, ind.phenotype, True)
    analyser.create_graph()
    return ind


def build_individual(filename, genome, grammar):
    """map genotype and save mesh file"""
    ind = Individual(None)
    generated_values = grammar.generate(genome)
    ind.set_values(generated_values)
    analyser = AZR.Analyser(filename, ind.phenotype, False)
    analyser.create_graph()
    analyser.create_mesh(filename)
    return ind


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


def onepoint_crossover(p, q, within_used=True):
    """Given two individuals, create two children using one-point
    crossover and return them."""
    # Get the chromosomes
    pc, qc = p.genome, q.genome
    # Uniformly generate crossover points. If within_used==True,
    # points will be within the used section.
    if within_used:
        maxp, maxq = p.used_codons, q.used_codons
    else:
        maxp, maxq = len(pc), len(qc)
    pt_p, pt_q = random.randint(1, maxp), random.randint(1, maxq)
    # Make new chromosomes by crossover: these slices perform copies
    if random.random() < CROSSOVER_PROBABILITY:
        c = pc[:pt_p] + qc[pt_q:]
        d = qc[:pt_q] + pc[pt_p:]
    else:
        c, d = pc, qc
    # Put the new chromosomes into new individuals
    return [Individual(c), Individual(d)]


def evaluate_fitness(individuals, grammar, fitness_function):
    """Perform the mapping and evaluate each individual"""
    for ind in individuals:
        generated_values = grammar.generate(ind.genome)
        ind.set_values(generated_values)
        if ind.phenotype != None:
            ind.evaluate(fitness_function)
        else:
            print "BROKEN PHENOTYPE"
            AZR.log_error(ind.genome, "genotype could not be mapped")


def step(parent_pop, fronts, grammar, selection, fitness_function, best_ever):
    """perform single iteration and return next generation"""
    #Select parents
    pop_size = len(parent_pop)
    parents = selection(parent_pop)
    #Crossover parents and add to the new population
    child_pop = []
    while len(child_pop) < GENERATION_SIZE:
        child_pop.extend(onepoint_crossover(*random.sample(parents, 2)))

    #Mutate the new population
    child_pop = list(MUT_OPERATOR(child) for child in child_pop)
    #Evaluate the fitness of the new population
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
    create_meshes(new_pop)
    return new_pop, fronts, best_ever


def create_meshes(population):
    """assign uid and creating meshes"""
    for idx, indiv in enumerate(population):
        indiv.uid = idx
        analyser = AZR.Analyser(indiv.uid, indiv.phenotype, True)
        analyser.create_graph()


def run_analysis(indiv):
    """create mesh and show it using bmpost"""
    analyser = AZR.Analyser(indiv.uid, indiv.phenotype)
    analyser.show_mesh()

def run_optimization(indiv,filename=None,button=True):
    optimizer = OPT.Optimizer(indiv.uid,indiv.phenotype)
    if filename == None:
        filename = indiv.uid
        optimizer.optimize_size(filename,button=True)
    else:
        optimizer.optimize_size(filename,button=True)

def search_loop(max_gens, individuals, grammar, selection, fit_func):
    """Loop over max generations"""
    #Evaluate initial population
    fronts = []
    evaluate_fitness(individuals, grammar, fit_func)
    best_ever = max(individuals)
    individuals.sort(reverse=True)
    print_stats(1, individuals)
    for generation in xrange(2, (max_gens + 1)):
        individuals, fronts, best_ever = step(
            individuals, fronts, grammar, selection, fit_func, best_ever)
        print_stats(generation, individuals)
        count_fronts(fronts)
        write_front(generation, fronts)
        write_pop(generation, individuals)
        save_pop(individuals)
    last = count_fronts(fronts)
    write_fronts(fronts[0:last])
    write_mesh(fronts[0:3], 'front')
    return individuals

def save_pop(population):
    #write last pop to file:
    filename = str(TIME_STAMP) + ".dat"
    savefile = open(filename, 'w')
    for indiv in population:
        savefile.write(str(indiv.uid) + ';' + str(indiv.fitness) + ';' + str(indiv.genome) + '\n')
    savefile.close()


#GE Properties
SAVE_BEST = False
CODON_SIZE = 100
ELITE_SIZE = 2
POPULATION_SIZE = 10
GENERATION_SIZE = 10
FRONT_FOLDER = "frontData"
GENERATIONS = 50
DEFAULT_FIT = 100000000000
MUTATION_PROBABILITY = 0.015
MUT_OPERATOR = int_flip_mutation
CROSSOVER_PROBABILITY = 0.7
GRAMMAR_FILE = "grammars/jon_pylon10.bnf"
FITNESS_FUNCTION = StructuralFitness()
IMG_COUNTER = 0
#create a timestamp
now = datetime.datetime.now()
hms = "%02d%02d%02d" % (now.hour, now.minute, now.second)
TIME_STAMP = (str(now.day) + "_" + str(now.month) + "_" + hms)


if __name__ == "__main__":
    from optparse import OptionParser
    import grammar as GRAMMAR

    parser = OptionParser(usage="if nothing is specified, it uses the default values specified in evolver class")
    parser.set_defaults(pop_size=POPULATION_SIZE, generations=GENERATIONS,
                        elite_size=ELITE_SIZE, mutation=MUTATION_PROBABILITY,
                        bnf_grammar=GRAMMAR_FILE, crossover=CROSSOVER_PROBABILITY)

    parser.add_option("-p", "--population", dest="pop_size",
                      help=" Number of individuals in the population")
    parser.add_option("-g", "--generations", dest="generations",
                      help="Number of iterations of the algorithm")
    parser.add_option("-e", "--elite_size", dest="elite_size",
                      help=" How many get copied to next generation")
    parser.add_option("-m", "--mutation", dest="mutation",
                      help="probability of mutation on a per-codon basis")
    parser.add_option("-c", "--crossover", dest="crossover",
                      help="probability of crossover")
    parser.add_option("-b", "--bnf_grammar", dest="bnf_grammar",
                      help="bnf grammar for mapping")
    parser.add_option("-f", "--front_folder", dest="front_folder",
                      help="output the fronts to this folder")
    opts, args = parser.parse_args()
#    if opts.bnf_grammar is None:
#        parser.print_help()
#        exit()

    POPULATION_SIZE = int(opts.pop_size)
    GENERATION_SIZE = int(opts.pop_size)
    GENERATIONS = int(opts.generations)
    ELITE_SIZE = int(opts.elite_size)
    MUTATION_PROBABILITY = float(opts.mutation)
    CROSSOVER_PROBABILITY = float(opts.crossover)
    GRAMMAR_FILE = opts.bnf_grammar
    if not opts.front_folder == None:
        FRONT_FOLDER = os.getcwd() + '/' + opts.front_folder
        print "writing fronts in folder:", FRONT_FOLDER
        if not os.path.isdir(FRONT_FOLDER):
            print "folder doesn't exist, creating folder"
            os.makedirs(FRONT_FOLDER)
    else:
        print "no front folder specified, using default"

    # Read grammar
    BNF_GRAMMAR = GRAMMAR.Grammar(GRAMMAR_FILE)
    # Create Individual
    INDIVIDUALS = initialise_population(POPULATION_SIZE)
    # Loop
    LAST_POP = search_loop(GENERATIONS, INDIVIDUALS, BNF_GRAMMAR, tournament_selection, FITNESS_FUNCTION)
    save_pop(LAST_POP)
