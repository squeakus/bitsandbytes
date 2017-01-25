#! /usr/bin/env python

# Femtastic Mesh generator
# Copyright (c) 2010 Jonathan Byrne, Erik Hemberg and James McDermott
# Hereby licensed under the GNU GPL v3.

#TODO#
# FIX RANGE IN BEAM NORMALISATION!
# build a more complicated grammar using all the meshing techniques
# use R for graphs
# build meshes of front
# fix crowding function
# fix "best"
# multiprocessor subprocesses
#remove defaultfit
# fix max value for normalisation

import sys, re, copy, random, math, time, multiprocessing
from mesh import Mesh

class Grammar(object):
    NT = "NT" # Non Terminal
    T = "T" # Terminal

    def __init__(self, file_name):
        if file_name.endswith("pybnf"):
            self.python_mode = True
        else:
            self.python_mode = False
        self.readBNFFile(file_name)

    def readBNFFile(self, file_name):
        """Read a grammar file in BNF format"""
        # <.+?> Non greedy match of anything between brackets
        NON_TERMINAL_PATTERN = "(<.+?>)"
        RULE_SEPARATOR = "::="
        PRODUCTION_SEPARATOR = "|"

        self.rules = {}
        self.non_terminals, self.terminals = set(), set()
        self.start_rule = None
        # Read the grammar file
        for line in open(file_name, 'r'):
            if not line.startswith("#") and line.strip() != "":
                # Split rules. Everything must be on one line
                if line.find(RULE_SEPARATOR):
                    lhs, productions = line.split(RULE_SEPARATOR)
                    lhs = lhs.strip()
                    if not re.search(NON_TERMINAL_PATTERN, lhs):
                        raise ValueError("lhs is not a NT:",lhs)
                    self.non_terminals.add(lhs)
                    if self.start_rule == None:
                        self.start_rule = (lhs, self.NT)
                    # Find terminals
                    tmp_productions = []
                    for production in [production.strip()
                                       for production in productions.split(PRODUCTION_SEPARATOR)]:
                        tmp_production = []
                        if not re.search(NON_TERMINAL_PATTERN, production):
                            self.terminals.add(production)
                            tmp_production.append((production, self.T))
                        else:
                            # Match non terminal or terminal pattern
                            # TODO does this handle quoted NT symbols?
                            for value in re.findall("<.+?>|[^<>]*", production):
                                if value != '':
                                    if not re.search(NON_TERMINAL_PATTERN, value):
                                        symbol = (value, self.T)
                                    else:
                                        symbol = (value, self.NT)
                                    tmp_production.append(symbol)
                        tmp_productions.append(tmp_production)
                    # Create a rule
                    if not lhs in self.rules:
                        self.rules[lhs] = tmp_productions
                    else:
                        raise ValueError("lhs should be unique", lhs)
                else:
                    raise ValueError("Each rule must be on one line")

    def generate(self, input, max_wraps=2):
        """Map input via rules to output. Returns output and used_input"""
        used_input = 0
        wraps = 0
        output = []

        unexpanded_symbols = [self.start_rule]
        while (wraps < max_wraps) and (len(unexpanded_symbols) > 0):
            # Wrap
            if used_input % len(input) == 0 and used_input > 0:
                wraps += 1
            # Expand a production
            current_symbol = unexpanded_symbols.pop(0)
            # Set output if it is a terminal
            if current_symbol[1] != self.NT:
                output.append(current_symbol[0])
            else:
                production_choices = self.rules[current_symbol[0]]
                # Select a production
                current_production = input[used_input % len(input)] % len(production_choices)
                # Use an input if there was more then 1 choice
                if len(production_choices) > 1:
                    used_input += 1
                # Derviation order is left to right(depth-first)
                unexpanded_symbols = production_choices[current_production] + unexpanded_symbols

        #Not completely expanded
        if len(unexpanded_symbols) > 0:
            return (None, used_input)

        output = "".join(output)
        if self.python_mode:
            output = self.python_filter(output)
        return (output, used_input)
        
class StructuralFitness():
    """Fitness function for testing generated mesh programs. """
    maximise = False # false = smaller is better
    def __init__(self,saveRun=False):
        self.saveRun = saveRun
        self.maxComp = 0
        self.maxTens = 0
        print "Using SLFFEA fitness function"

    def __call__(self, program):
        mesh = Mesh(program,self.saveRun)
        self.assign_material(mesh)
        mesh.test_mesh()
        fitness,beams = self.calculate_fitness(mesh)
        return fitness, beams

    def assign_material(self,mesh):
        width = 100
        height = 200
        emod = 10000000000
        dens = 5300
        mesh.set_material(Width=width,Height=height,Emod=emod,Density=dens)

    
    def calculate_fitness(self, mesh):
        beams = len(mesh.edgeList)
        maxComp = max(mesh.XXStress['maxComp'],mesh.XYStress['maxComp'],mesh.ZXStress['maxComp'])
        if self.maxComp < maxComp:
            self.maxComp = maxComp
        maxTens = max(mesh.XXStress['maxTens'],mesh.XYStress['maxTens'],mesh.ZXStress['maxTens'])
        if self.maxTens < maxTens:
            self.maxTens = maxTens
        #print "max comp:",self.maxComp," max tens:",self.maxTens
        x,y,z = mesh.XXStress['failTens'],mesh.XYStress['failTens'],mesh.ZXStress['failTens'] 
        failTotal = x + y + z
        avrComp = mesh.XXStress['avrComp']+mesh.XYStress['avrComp']+mesh.ZXStress['avrComp']
        avrCompNorm = self.normalize(avrComp,[1,1000],[0,maxComp])
        avrTens = mesh.XXStress['avrTens']+mesh.XYStress['avrTens']+mesh.ZXStress['avrTens']
        avrTensNorm = self.normalize(avrTens,[1,1000],[0,maxTens])
        fitness = beams+avrCompNorm+avrTensNorm
        #print "beams: ",beams," avrComp:",avrComp," avrTens:",avrTens," ACNorm:",avrCompNorm," ATNorm:",avrTensNorm," failed: ",failTotal
        return fitness, beams

    def normalize(self,value,newRange,oldRange):
        if value > oldRange[1]:
            value = oldRange[1]
        normalized = (newRange[0]+(value-oldRange[0])*(newRange[1]-newRange[0]))/oldRange[1]-oldRange[0] 
        return normalized

class Individual(object):
    """A GE individual"""
    def __init__(self, genome, length=100):
        if genome == None:
            self.genome = [random.randint(0, CODON_SIZE)
                           for i in xrange(length)]
        else:
            self.genome = genome
        self.bad = default_fitness(FITNESS_FUNCTION.maximise)
        self.phenotype = None
        self.rank = None
        self.distance = None
        self.beamTotal = self.bad
        self.stresses = self.bad
        self.used_codons = 0
        self.fitness = [int(self.stresses),int(self.beamTotal)]

    def __lt__(self, other):
        if FITNESS_FUNCTION.maximise:
            return self.fitness < other.fitness
        else:
            return other.fitness < self.fitness

    def __str__(self):
        return ("Individual: " +
                str(self.phenotype) + "; " + str(self.fitness))

    def evaluate(self, fitness):
        self.stresses,self.beamTotal = fitness(self.phenotype)
        self.fitness = [int(self.stresses),int(self.beamTotal)]

    def dominates(self, other):
        dominated = False
        won,lost = False, False
        for val in range(len(self.fitness)):
            if self.fitness[val] < other.fitness[val]:
                won = True
            elif self.fitness[val] > other.fitness[val]:
                lost = True
        if won == True and lost == False:
            dominated = True
        return dominated

def initialise_population(size=10):
    """Create a popultaion of size and return"""
    return [Individual(None) for cnt in xrange(size)]

def print_stats(generation, individuals): 
    global TIME
    def ave(values):
        return float(sum(values))/len(values)
    def std(values, ave):
        return math.sqrt(float(sum((value-ave)**2 for value in values))/len(values))

    newTime = time.time()
    genTime = newTime - TIME
    TIME = newTime
    ave_beam= ave([i.beamTotal for i in individuals if i.phenotype is not None])
    std_beam = std([i.beamTotal for i in individuals if i.phenotype is not None], ave_beam)
    ave_fit = ave([i.fitness[0] for i in individuals if i.phenotype is not None])
    std_fit = std([i.fitness[0] for i in individuals if i.phenotype is not None], ave_fit)
    ave_used_codons = ave([i.used_codons for i in individuals
                           if i.phenotype is not None])
    std_used_codons = std([i.used_codons for i in individuals
                           if i.phenotype is not None], ave_used_codons)
    print("Gen:%d best:%s beams:s:%d ave:%.1f+-%.1f Used:%.1f+-%.1f tt:%.2f beams:%d+-%.1f" % (generation,individuals[0].fitness,individuals[0].beamTotal,ave_fit,std_fit,ave_used_codons,std_used_codons,genTime,ave_beam,std_beam))

    if SAVE_BEST:
        print "saving best individual"
        bestMesh = Mesh(individuals[0].phenotype)
        filename = 'xxx.'+str(generation)
        bestMesh.create_mesh(True,filename)

def default_fitness(maximise=False):
    if maximise:
        return -DEFAULT_FIT
    else:
        return DEFAULT_FIT

#takes a population and returns a list of fronts
def fast_nondominated_sort(pop):
    fronts = []
    domList = dict()
    for A in pop:
        domList[hash(A)] = []
        A.domCount = 0
        for B in pop:
            if A.dominates(B):
                domList[hash(A)].append(B)
            elif B.dominates(A):
                A.domCount += 1
        if A.domCount == 0:
            A.rank = 1
            fronts.append([]) # add new front
            fronts[0].append(A) # add to first front

    i = 0 #front counter
    while len(fronts[i]) != 0:
        newPop = []
        for A in fronts[i]:
            for B in domList[hash(A)]:
                B.domCount -= 1
                if B.domCount == 0:
                    B.rank = i + 1
                    newPop.append(B)
        i += 1
        fronts.append([])
        fronts[i] = newPop
    print "non empty fronts: ",i
    return fronts

def crowded_comparasion_operator(x,y):
    smallestFirst = False
    if x.rank < y.rank:
        return +1
    elif (x.rank == y.rank):
        if x.distance > y.distance:
            return -1
        else:
            return +1

#calulates distances between individuals
def crowding_distance_assignment(pop):
    popSize = len(pop)
    for indiv in pop:
        indiv.distance = 0
    #assign cumulative distance to each individual
    for val in range(len(pop[0].fitness)):
        pop.sort(lambda x,y: cmp(x.fitness[val], y.fitness[val]))
        
        #always include boundary points
        pop[0].distance = 9999
        pop[popSize-1].distance = 9999
        for i in range(2, popSize-1):
            pop[i].distance += (pop[i+1].fitness[val]-pop[i-1].fitness[val])
        #send back a sorted pop
    pop.sort(crowded_comparasion_operator)

def write_front(idx,fronts):   
    file = open("./frontData/Gen" + repr(idx).zfill(3) + ".dat", "w")
    for front in fronts:
        for indiv in front:
            t = repr(indiv.fitness[0]).rjust(5)+" "+repr(indiv.fitness[1]).rjust(5)+"\n"
            file.write(t)
    file.close()

def write_fronts(fronts):
    for idx,front in enumerate(fronts):
        if len(front) > 0:               
            file = open("./frontData/Front" + repr(idx).zfill(3) + ".dat", "w")
            for indiv in front:
                t = repr(indiv.fitness[0]).rjust(5)+" "+repr(indiv.fitness[1]).rjust(5)+"\n"
                file.write(t)
            file.close()

def write_mesh(fronts):
    counter=0
    for front in fronts:
        for indiv in front:
            mesh = Mesh(str(indiv.phenotype))
            mesh.set_material()
            filename = "xxx."+str(counter)
            print "filename: ",filename
            mesh.create_mesh(True,filename)
            counter += 1

def int_flip_mutation(individual):
    """Mutate the individual by randomly chosing a new int with
    probability p_mut. Works per-codon, hence no need for
    "within_used" option."""
    for i in xrange(len(individual.genome)):
        if random.random() < MUTATION_PROBABILITY:
            individual.genome[i] = random.randint(0,CODON_SIZE)
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
    # Perform the mapping for each individual
    for ind in individuals:
        count = multiprocessing.cpu_count()
        #print "cpu count: ",count
        #pool = multiprocessing.Pool(processes=count)
        #print str(pool)
        ind.phenotype, ind.used_codons = grammar.generate(ind.genome)
        if ind.phenotype != None:
            ind.evaluate(fitness_function)

def interactive_evaluate_fitness(individuals, grammar, callback):
    # perform mapping, set dummy fitness
    evaluate_fitness(individuals, grammar, lambda x: 0.0)
    fitness_values = callback()
    for i, individual in enumerate(individuals):
        if individual.phenotype != None:
            individual.fitness = fitness_values[i]

def generational_replacement(new_pop, individuals):
    individuals.sort(reverse=True)
    for ind in individuals[:ELITE_SIZE]:
        new_pop.append(copy.copy(ind))
    new_pop.sort(reverse=True)
    return new_pop[:GENERATION_SIZE]

def steady_state_replacement(new_pop, individuals):
    individuals.sort(reverse=True)
    individuals[-1] = max(new_pop + individuals[-1:])
    return individuals

def step(parent_pop,front, grammar, replacement, selection, fitness_function, best_ever):
    #Select parents
    pop_size = len(parent_pop)
    #parents = selection(parent_pop)
    parents = parent_pop
    #Crossover parents and add to the new population
    child_pop = []
    while len(child_pop) < GENERATION_SIZE:
        child_pop.extend(onepoint_crossover(*random.sample(parents, 2)))
    #Mutate the new population
    child_pop = list(map(int_flip_mutation, child_pop))
    #Evaluate the fitness of the new population
    evaluate_fitness(child_pop, grammar, fitness_function)

    #run fast non-dominated sort on child+parent pop
    total_pop = []
    front = []
    total_pop.extend(parent_pop)
    total_pop.extend(child_pop)
    front = fast_nondominated_sort(total_pop)

    #assign distance and append fronts to new population
    new_pop = []
    i = 0 #front counter  
    while len(new_pop)+len(front[i]) <= pop_size:
        crowding_distance_assignment(front[i])
        new_pop.extend(front[i])
        i += 1
    # filling up pop with the final front
    crowding_distance_assignment(front[i])
    new_pop.extend(front[i][0 : pop_size-len(new_pop)])
    best_ever = max(best_ever, max(new_pop))

    #print "Total pop"
    #for idx,indiv in enumerate(new_pop):
    #    print "indiv:",idx," rank: ",indiv.rank, " dist: ",indiv.distance

    return new_pop, front, best_ever

def search_loop(max_generations, individuals, grammar, replacement, selection, fitness_function):
    """Loop over max generations"""
    #Evaluate initial population
    front = []
    evaluate_fitness(individuals, grammar, fitness_function)
    best_ever = max(individuals)
    individuals.sort(reverse=True)
    print_stats(1,individuals)
    for generation in xrange(2,(max_generations+1)):
        individuals, front, best_ever = step(
            individuals,front, grammar, replacement, selection, fitness_function, best_ever)
        print_stats(generation, individuals)
        write_front(generation,front[0:2])
    write_fronts(front[0:8])
    write_mesh(front[0:3])
    return best_ever

#GE Properties
TIME = time.time()
SAVE_BEST = False
SAVE_RUN = False
CODON_SIZE = 100
ELITE_SIZE = 1
POPULATION_SIZE = 100
GENERATION_SIZE = 100
GENERATIONS = 10
DEFAULT_FIT = 100000
MUTATION_PROBABILITY = 0.15
CROSSOVER_PROBABILITY = 0.7
GRAMMAR_FILE =  "./grammars/pylon.bnf"
FITNESS_FUNCTION = StructuralFitness(SAVE_RUN)
IMG_COUNTER = 0

# Run program
def mane():
    # Read grammar
    bnf_grammar = Grammar(GRAMMAR_FILE)
    # Create Individuals
    individuals = initialise_population(POPULATION_SIZE)
    # Loop
    best_ever = search_loop(GENERATIONS, individuals, bnf_grammar, generational_replacement, tournament_selection, FITNESS_FUNCTION)
    bestMesh = Mesh(str(best_ever.phenotype))
    bestMesh.set_material()
    bestMesh.show_mesh()
    
if __name__ == "__main__":
     import getopt
     try:
          #FIXME help option
          print(sys.argv)
          opts, args = getopt.getopt(sys.argv[1:], "p:g:e:m:x:b:f:", ["population", "generations", "elite_size", "mutation", "crossover", "bnf_grammar", "fitness_function"])
     except getopt.GetoptError, err:
          print(str(err))
          #FIXME usage
          sys.exit(2)
     for o, a in opts:
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
          elif o in ("-f", "--fitness_function"):
               FITNESS_FUNCTION = eval(a)
          else:
               assert False, "unhandeled option"
     mane()
