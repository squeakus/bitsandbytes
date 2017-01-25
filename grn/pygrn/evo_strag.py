import random, graph

class Evostrategy:
    """A 1+1 evolutionary strategy using the 1/5th rule"""
    def __init__(self):
        self.codon_size = 1000
        self.mut_rate = 0.1
        self.delta = 0.001
        self.adaptive = False
        self.adapt_mod = 100 #eval mut_rate every 100 generations
        self.target = [1 for _ in range(0, self.codon_size)]

    def create_indiv(self):
        """initialise a binary individual"""
        indiv = [random.randint(0, 1) for _ in range(0, self.codon_size)]
        return indiv

    def mutate(self, indiv, mut_rate):
        """mutate each codon with a given probability"""
        mutant = list(indiv)
        for i in range(len(mutant)):
            if random.random() < mut_rate:
                mutant[i] = random.randint(0, 1)
        return mutant
                
    def onemax_fitness(self, indiv):
        """sum the number of ones in the genome"""
        fitness = sum(indiv[i] == self.target[i]
                      for i in range(0, len(self.target)))
        return fitness

    def evolve(self, generations):
        """mutate and evaluate"""
        parent = self.create_indiv()
        mut_rate, delta = self.mut_rate, self.delta
        run = []
        success_mut = 0

        for generation in range(generations):
            child = self.mutate(parent, mut_rate)
            parent_fit = self.onemax_fitness(parent)
            child_fit = self.onemax_fitness(child)

            if child_fit > parent_fit:
                parent = child
                success_mut += 1
            run.append(max(parent_fit, child_fit))

            # adapt mutation rate
            if self.adaptive:
                if generation % self.adapt_mod == 0:
                    success_rate = float(success_mut) / self.adapt_mod
                    # 1/5th rule
                    if success_rate > 0.2:
#                        delta = delta * 2
                        mut_rate = mut_rate * 2
                    else:
                        mut_rate = mut_rate / 2
#                        delta = delta / 0.2
#                    mut_rate = random.gauss(mut_rate, delta)
                    success_mut = 0

                # make sure mut_rate is within boundaries
                if mut_rate < 0.0001: mut_rate = 0.0001
                if mut_rate > 0.5: mut_rate = 0.5
        return run

def main():
    results = []
    evo = Evostrategy()
    
    for _ in range(100):
        run = evo.evolve(1000)
        results.append(run)
        
    graph.plot_3d(results)
    graph.plot_ave(results)

if __name__ == "__main__":
    main()
