from nsga import *

class Individual(object):
    def __init__(self, fitness):
        self.fitness = fitness

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


fitnesses = [[4,5],[5,4],[7,6],[3,3],[2,3],[6,6],[1,5],[6,3],[6,7],[8,3]]

population = []
for fitness in fitnesses:
    population.append(Individual(fitness))

    
fast_nondominated_sort(population)
