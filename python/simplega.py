"""
This is the simplest GA implementation I can think of. It is a bit
verbose but that is to make it clear what is going on. The problem
that we are trying to solve is the "onemax" problem. Basically our
individuals consist of ones and zeros and we are trying to find an
individual consisting entirely of ones.

this is what a random individual looks like:
[0,0,0,1,1,0,1,0,1,1,1,0,0,0,0,0,1,0,0]
you compute the fitness by summing the ones, the individual above has
a fitness of 7 as it has 7 ones in it.

and this is what we are looking for:
[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

There are 5 parts to a Genetic algorithm:
1: initialisation (create the first generation of random guesses)
2: evaluation (assign a fitness value to the individuals)
3: selection (stochastically select the best individuals from the population)
4: mutation (flip a bit in the individual with a certain probability)
5: crossover (split two good individuals in half and recombine them)

repeat steps 2 to 5 until you get the answer. simple as.
"""
import random

def main():
    """
    Works exactly as described above, tournament select, mutate and
    crossover are called in the select method.
    """
    generations = 5 #maximum number of generations
    pop_size = 10 # number of individuals in the population
    genome_length = 10 # the length of the individual's array
    tournament_size = 3 # the number of individuals in a tournament
    mut_prob = 0.01 # probability of flipping a bit in an individual

    population = create_pop(pop_size, genome_length)
    # lets start grinding!
    for gen in range(generations):
        evaluate(population)

        print "Generation:", gen, "best:", population[0]['fitness']
        # if you have found the correct answer then quit
        if population[0]['fitness'] == genome_length: exit()
        # otherwise create the next generation
        population = select(population, tournament_size, mut_prob)        

def create_genome(genome_length):
    """
    The genome consists of an array of ones and zeros
    """
    genome = []
    for _ in range(genome_length):
        genome.append(random.randint(0, 1))
    return genome
    
def create_pop(pop_size, genome_length):
    """
    each individual in the population consists of a genome and fitness value
    """
    population = []
    for _ in range(pop_size):
        individual = {'genome': create_genome(genome_length), 'fitness':0}
        population.append(individual)
    return population

def evaluate(population):
    """
    calculate the fitness for each individual by summing the genome
    then sort the population by fitness so the best is first
    """
    for indiv in population:
        indiv['fitness'] = sum(indiv['genome'])
    population.sort(key=lambda k: k['fitness'], reverse=True)
    return population

def mutate(indiv, probability):
    """
    Iterate over the genes in the genome and change with a given
    probability.
    """
    for i in range(len(indiv['genome'])):
        if random.random() < probability:
            indiv['genome'][i] = random.randint(0, 1)

def crossover(winners):
    """
    given two individuals, chop their genomes in half and create two
    new offspring.
    """
    
    genome_a, genome_b = winners[0]['genome'], winners[1]['genome']
    mid = len(genome_a)
    genome_c = genome_a[:mid] + genome_b[mid:]
    genome_d = genome_b[:mid] + genome_a[mid:]
    child_a = {'genome': genome_c, 'fitness':0}
    child_b = {'genome': genome_d, 'fitness':0}
    return child_a, child_b
    
def tournament(population, tournament_size):
    """
    randomly sample from the population to fill a tournament, then
    take the best two from the tournament.
    """
    tourn = []
    for _ in range(tournament_size):
        tourn.append(random.choice(population))
        
    tourn.sort(key=lambda k: k['fitness'], reverse=True)
    winners = tourn[:2]
    return winners
    
def select(population, tournament_size, mut_prob):
    """
    select individuals from the old population using tournament select
    mutate them, cross them over  and add them to the new population
    """
    new_pop = []
    while len(new_pop) < len(population):
        winners = tournament(population, tournament_size)

        for winner in winners:
            mutate(winner, mut_prob)

        child_a, child_b = crossover(winners)
        new_pop.append(child_a)
        new_pop.append(child_b)
    return new_pop

def print_pop(population):
    for indiv in population:
        print 'genome:', indiv['genome'], 'fitness', indiv['fitness']

if __name__ == '__main__':
    main()
