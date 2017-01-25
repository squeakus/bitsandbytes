"""Simple Evolutionary strategy algorithm"""
import random, graph

class Evostrategy:
    """A 1+1 evolutionary strategy using the 1/5th rule"""    
    def __init__(self, genome_size):
        self.gen_count = 0
        self.genome_size = genome_size
        self.mut_rate = 0.01
        self.success_mut = 0
        self.parent = {'genome':self.create_indiv(),'fitness':0}
        self.adaptive = True
        self.adapt_mod = 1 #eval mut_rate every 10 generations
        self.target = [1 for _ in range(0, self.genome_size)]

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
        if self.gen_count % self.adapt_mod == 0:
            success_rate = float(self.success_mut) / self.adapt_mod
            # 1/5th rule
            if success_rate > 0.2:
                self.mut_rate = self.mut_rate * 2
            else:
                self.mut_rate = self.mut_rate / 2
            self.success_mut = 0
            # make sure self.mut_rate is within boundaries
            if self.mut_rate < 0.001: self.mut_rate = 0.001
            if self.mut_rate > 0.5: self.mut_rate = 0.5
        
    def onemax_fitness(self, indiv):
        """sum the number of ones in the genome"""
        fitness = sum(indiv[i] == self.target[i]
                      for i in range(0, len(self.target)))
        return fitness
            
    def evolve(self, generations):
        """mutate and evaluate"""
        parent = self.create_indiv()
        run = {'bestfit':[],
               'mutrate':[]}
        success_mut = 0

        for generation in range(generations):
            child = self.mutate(parent, self.mut_rate)
            parent_fit = self.onemax_fitness(parent)
            child_fit = self.onemax_fitness(child)

            if child_fit > parent_fit:
                parent = child
                success_mut += 1

            # adapt mutation rate
            if self.adaptive:
                self.adapt_mutation()

            run['bestfit'].append(max(parent_fit, child_fit))
            run['mutrate'].append(self.mut_rate)

        return run


    def iterate(self, child):
        self.gen_count += 1
        if self.parent['fitness'] < child['fitness']:
            self.success_mut += 1
            self.parent = child
        mutated_genome = self.mutate(self.parent['genome'])
        new_child = {'genome':mutated_genome, 'fitness':0}
        return new_child

        
def main():
    best_list, mut_list = [], []
    evo = Evostrategy(5000)
    fitness = evo.onemax_fitness(evo.parent['genome'])
    indiv = {'genome':evo.parent['genome'], 'fitness':fitness}
    child = evo.iterate(indiv)

    for i in range(100):
        child['fitness'] = evo.onemax_fitness(child['genome'])
        child = evo.iterate(child)
        print evo.parent['fitness']
        if evo.parent['fitness'] == 5000:
            break
        
        if evo.adaptive:
            evo.adapt_mutation()
        best_list.append(evo.parent['fitness'])
        mut_list.append(evo.mut_rate)

    graph.plot_2d([best_list], 'bestfit')
    graph.plot_2d([mut_list], 'mutrate')
    # for _ in range(1000):
    #     run = evo.evolve(1000)
    #     best_list.append(run['bestfit'])
    #     mut_list.append(run['mutrate'])

    # graph.plot_2d(best_list, 'bestfit')
    # graph.plot_ave(best_list, 'bestfit')

    # graph.plot_2d(mut_list, 'mutrate')
    # graph.plot_ave(mut_list, 'mutrate')

    
if __name__ == "__main__":
    main()
