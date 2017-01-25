"""Simple Evolutionary strategy algorithm"""
import random, graph

class Evostrategy:
    """A population based evolutionary strategy using the 1/5th rule"""    
    def __init__(self, genome_size, pop_size):
        self.genome_size = genome_size
        self.pop_size = pop_size
        self.pop = []
        self.initialise_pop()
        self.gen_count = 0
        self.mut_rate = 0.01
        self.success_mut = 0
        self.adaptive = True
        self.target = [1 for _ in range(0, self.genome_size)]

    def initialise_pop(self):
        for _ in range(self.pop_size):
            self.pop.append({'genome':self.create_indiv(),
                             'fitness':0,
                             'testfit':0})

    def create_indiv(self):
        """initialise a binary individual"""
        indiv = [random.randint(0, 1) for _ in range(0, self.genome_size)]
        return indiv

    def mutate(self,indiv):
        """mutate each codon with a given probability"""
        mutant = list(indiv)
        for i in range(len(mutant)):
            if random.random() < self.mut_rate:
                mutant[i] = random.randint(0, 1)
        return mutant

    def adapt_mutation(self):
        success_rate = float(self.success_mut) / self.pop_size
        # 1/5th rule
        if success_rate > 0.2:
            self.mut_rate = self.mut_rate * 2
        else:
            self.mut_rate = self.mut_rate / 2
        self.success_mut = 0
        # make sure self.mut_rate is within boundaries
        if self.mut_rate < 0.0005: self.mut_rate = 0.0005
        if self.mut_rate > 0.5: self.mut_rate = 0.5
                
    def onemax_fitness(self, indiv):
        """sum the number of ones in the genome"""
        fitness = sum(indiv[i] == self.target[i]
                      for i in range(0, len(self.target)))
        return fitness

    def iterate(self, children):
        self.gen_count += 1
        self.pop.sort(key=lambda k: k['fitness'])
        children.sort(key=lambda k: k['fitness'])
        children.reverse()

        for child in children:
            for idx, parent in enumerate(self.pop):
                if parent['fitness'] < child['fitness']:
                    self.success_mut += 1      
                    parent['genome'] = child['genome']
                    parent['fitness'] = child['fitness']
                    parent['testfit'] = child['testfit']
                    break
                
        self.pop.sort(key=lambda k: k['fitness'])
        children = []
        for parent in self.pop:
            mutated_genome = self.mutate(parent['genome'])
            children.append({'genome':mutated_genome,
                             'fitness':0,
                             'testfit':0})
        return children
        
def main():
    best_list, mut_list = [], []
    evo = Evostrategy(5000, 100)
    children = evo.iterate(evo.pop)

    for i in range(10):
        for child in children:
            child['fitness'] = evo.onemax_fitness(child['genome'])
        children = evo.iterate(children)

        if evo.adaptive:
            evo.adapt_mutation()

        best_list.append(evo.pop[-1]['fitness'])
        mut_list.append(evo.mut_rate)

if __name__ == "__main__":
    main()
