#! /usr/bin/env python

# evolver: does the evolvy bit
# Copyright (c) 2010 Jonathan Byrne, Erik Hemberg and James McDermott
# Hereby licensed under the GNU GPL v3.

#TODO#
# fix structures with missing last beam
# replace lists with graph itself
# hardcoded max in normalisation
# multiprocessor subprocesses
#remove defaultfit
# fix max value for normalisation

import sys 
import copy
import random
import math
import time
import graph
import difflib
from geometry import pt_plus_pt, interpolate, bezier_form
from grammar import Grammar
from math import exp, sin, cos, pi
random.seed(10)


class StructuralFitness():
    """Fitness function for testing generated mesh programs. """

    def __init__(self):
        self.maxComp = 9000000
        self.maxTens = 3000000

    def __call__(self):
        fitness, beams = self.calculate_fitness()
        return fitness, beams

    def calculate_fitness(self):
        fitness = 100000
        beams = 100000
        return fitness, beams

    def normalize(self, value, newRange, oldRange):
        if value > oldRange[1]:
            value = oldRange[1]
        normalized = ((newRange[0] + (value - oldRange[0])
                      * (newRange[1] - newRange[0]))
                      / oldRange[1] - oldRange[0])
        return normalized


class Individual(object):
    """A GE individual"""

    def __init__(self, genome, length=150):
        if genome == None:
            self.genome = [random.randint(0, CODON_SIZE)
                           for _ in xrange(length)]
        else:
            self.genome = copy.deepcopy(genome)
        self.bad = default_fitness()
        self.UID = None
        self.phenotype = None
        self.derivation_tree = None
        self.rank = None
        self.distance = None
        self.beamTotal = self.bad
        self.stresses = self.bad
        self.used_codons = 0
        self.codon_list = []
        self.codonList = []
        self.fitness = [int(self.stresses), int(self.beamTotal)]

    def __lt__(self, other):
        return other.fitness < self.fitness

    def __str__(self):
        return ("Individual UID: " + str(self.UID) + "; "
                + str(self.fitness) + ";" + str(self.genome))

    def evaluate(self, fitness):
        self.stresses, self.beamTotal = fitness()
        self.fitness = [int(self.stresses), int(self.beamTotal)]

    def set_values(self, values):
        self.phenotype = values['phenotype']
        self.used_codons = values['used_codons']
        self.codon_list = values['codon_list']
        self.derivation_tree = values['derivation_tree']


##########################graph methods#########################
def create_graph(UID, program, name='indiv'):
    myGraph = None
    nodeList = []
    edgeList = []

    myGraph = eval_or_exec(program)
    for node in myGraph.nodes():
        xyz = myGraph.get_node_data(node)
        label = myGraph.node[node]['label']
        node = {'id': str(node), 'x': xyz[0], 'y': xyz[1],
                'z': xyz[2], 'label': label}
        nodeList.append(node)
    for idx, edge in enumerate(myGraph.edges_iter()):
        edge = {'id': str(idx), 'ptA': str(edge[0]),
                'ptB': str(edge[1])}
        edgeList.append(edge)

    if name == 'indiv':
        filename = "population/indiv." + str(UID) + ".mesh"
    else:
        filename = name + '.mesh'
    mesh = open(filename, 'w')
    mesh.write("MeshVersionFormatted 1\nDimension\n3 \n")
    mesh.write("Vertices\n" + str(len(nodeList)) + " \n")
    for node in nodeList:
        mesh.write(str(node['x']) + " " + str(node['y'])
                   + " " + str(node['z']) + " 0  \n")
    mesh.write("Edges\n" + str(len(edgeList)) + " \n")
    for edge in edgeList:
        ptA, ptB = int(edge['ptA']), int(edge['ptB'])
        mesh.write(str(ptA + 1) + " " + str(ptB + 1) + " 0 \n")
    mesh.write("End\n")
    mesh.close()
    return myGraph


def graph_distance(graph_a, graph_b):
    """calculates fitness value by recording distance between nodes
    on the largest graph with the nearest node on the smaller graph"""
    if graph_a.size() > graph_b.size():
        bigger, smaller = graph_a, graph_b
    else:
        bigger, smaller = graph_b, graph_a

    edge_diff = bigger.number_of_edges() - smaller.number_of_edges()

    total_distance = 0
    for node_id in bigger.node:
        pos = bigger.node[node_id]['xyz']
        distance = smaller.nearest_node(pos[0], pos[1], pos[2])
        total_distance  += distance
    total_distance = round(total_distance, 0)
    return total_distance + edge_diff


################utility functions##############
def eval_or_exec(s):
    s = python_filter(s)
    try:
        retval = eval(s)
    except SyntaxError:
        exec(s)
        retval = XXXeval_or_exec_outputXXX
    return retval


def python_filter(txt):
    # Create correct python syntax. We use {} to track indentation,
    # which is not ideal because of the clash with dict literals.
    counter = 0
    if txt == None:
        return 0
    for char in txt:
        if char == "{":
            counter += 1
        elif char == "}":
            counter -= 1
        tabstr = "\n" + "  " * counter
        if char == "{" or char == "}":
            txt = txt.replace(char, tabstr, 1)
    txt = "\n".join([line for line in txt.split("\n")
                     if line.strip() != ""])
    return txt


################ GA functions##############
def initialise_population(size=10):
    """Create a popultaion of size and return"""
    return [Individual(None) for _ in xrange(size)]


def print_stats(generation, individuals):
    global TIME

    def ave(values):
        return float(sum(values)) / len(values)

    def std(values, ave):
        return math.sqrt(float(sum((value - ave) ** 2 for value in values))
                         / len(values))

    newTime = time.time()
    genTime = newTime - TIME
    TIME = newTime
    ave_fit = ave([i.fitness[0] for i in individuals
                   if i.phenotype is not None])
    std_fit = std([i.fitness[0] for i in individuals
                   if i.phenotype is not None], ave_fit)
    ave_used_codons = ave([i.used_codons for i in individuals
                           if i.phenotype is not None])
    std_used_codons = std([i.used_codons for i in individuals
                           if i.phenotype is not None], ave_used_codons)
    print("Gen:%d best:%s beams:s:%d ave:%.1f+-%.1f Used:%.1f+-%.1f tt:%.2f"
          % (generation, individuals[0].fitness, individuals[0].beamTotal,
             ave_fit, std_fit, ave_used_codons, std_used_codons, genTime))

    if SAVE_BEST:
        print "saving best individual"
        create_graph(0, individuals[0].phenotype)


def default_fitness():
    return DEFAULT_FIT


def mutate_type(individuals, grammar):
    # rebuild top individual
    cnt = 1
    generatedValues = grammar.generate(individuals[0].genome)
    individuals[0].set_values(generatedValues)
    original = create_graph(individuals[0].UID, individuals[0].phenotype)
    result = {'idx': 0, 'rule_change': "original", 'distance': 0}
    nodal_images = []
    struct_images = []
    nodal_images.append(result)
    struct_images.append(result)
    for codon in individuals[0].codon_list:
        prods, idx = codon['prods'], codon['idx']

        for i in range(prods):
            #print "indiv:", cnt, "codon:", idx, "prod:", i, "of", prods
            individuals[cnt] = Individual(individuals[0].genome)
            individuals[cnt].UID = cnt
            individuals[cnt].genome[idx] = individuals[cnt].genome[idx] + i
            generatedValues = grammar.generate(individuals[cnt].genome)
            individuals[cnt].set_values(generatedValues)
            new_graph = create_graph(individuals[cnt].UID,
                                     individuals[cnt].phenotype)
            distance = graph_distance(original, new_graph)
            if distance > 0:
                if codon['rule_type'] == 'nodal':
                    rule_change = ("nodal: " + str(codon['rule'])
                                   + " = " + str(codon['productions'][i]))
                    result = {'idx': cnt, 'rule_change': rule_change,
                              'distance': distance}
                    nodal_images.append(result)
                elif codon['rule_type'] == 'struct':
                    rule_change = ("struct: " + str(codon['rule'])
                                   + " = " + str(codon['productions'][i]))
                    result = {'idx': cnt, 'rule_change': rule_change,
                              'distance': distance}
                    struct_images.append(result)
                cnt += 1
    result = {'individuals': individuals, 'nodal_images': nodal_images,
              'struct_images': struct_images}
    return result


def mutate_codon(individuals, grammar, idx):
    # rebuild top individual
    generatedValues = grammar.generate(individuals[0].genome)
    individuals[0].set_values(generatedValues)
    create_graph(individuals[0].UID, individuals[0].phenotype)
    for codon in individuals[0].codon_list:
        if idx == codon['idx']:
            prods = codon['prods']

    #generate all variations
    for i in range(1, prods):
        individuals[i] = Individual(individuals[0].genome)
        individuals[i].UID = i
        individuals[i].genome[idx] = individuals[i].genome[idx] + i
        generatedValues = grammar.generate(individuals[i].genome)
        individuals[i].set_values(generatedValues)
        create_graph(individuals[i].UID, individuals[i].phenotype)

    #for i in range(prods):
        #print "after", individuals[i].genome[:30]
        #print "pheno:", individuals[i].phenotype
    return individuals


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
        generatedValues = grammar.generate(ind.genome)
        if ind.genome != None:
            ind.set_values(generatedValues)
            ind.evaluate(fitness_function)


def create_meshes(population):
    #assign UID and creating meshes
    for idx, indiv in enumerate(population):
        if not indiv.genome == None:
            indiv.UID = idx
            create_graph(indiv.UID, indiv.phenotype)


#GE Properties
TIME = time.time()
SAVE_BEST = False
CODON_SIZE = 100
ELITE_SIZE = 1
POP_SIZE = 1
GENERATION_SIZE = 1
GENERATIONS = 1
DEFAULT_FIT = 100000
MUTATION_PROBABILITY = 0.015
CROSSOVER_PROBABILITY = 0.0
GRAMMAR_FILE = "grammars/support.bnf"
FITNESS_FUNCTION = StructuralFitness()
IMG_COUNTER = 0


# Run program
def mane():
    # Read grammar
    bnf_grammar = Grammar(GRAMMAR_FILE)
    # Create Individual
    individuals = initialise_population(POP_SIZE)
    evaluate_fitness(individuals, bnf_grammar, FITNESS_FUNCTION)


if __name__ == "__main__":
    import getopt
    try:
        print(sys.argv)
        opts, args = getopt.getopt(sys.argv[1:], "p:g:e:m:x:b:",
                                   ["population", "generations",
                                    "elite_size", "mutation",
                                    "crossover", "bnf_grammar"])
    except getopt.GetoptError, err:
        print(str(err))

        sys.exit(2)
    for o, a in opts:
        if o in ("-p", "--population"):
            POP_SIZE = int(a)
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
        else:
            assert False, "unhandeled option"
    mane()
