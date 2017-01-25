import random, graph

def gaussian(ave, dev):
    """return a mutation change from a normal distribution"""
    val = random.gauss(ave, dev)
    return val

class Evostrategy:
    """A 1+1 evolutionary strategy using the 1/5th rule"""
    def __init__(self):
        self.generations = 10
        self.codon_size = 256
        self.mut_rate = 0.001
        self.delta = 0.001
        self.adaptive = False
        self.eval_mod = 100
        self.target = [1 for _ in range(0, self.codon_size)]

    def create_indiv(self):
        """initialise a binary individual"""
        indiv = [random.randint(0, 1) for _ in range(0, self.codon_size)]
        return indiv

    def mutate(self, indiv):
        
        mutant = list(indiv)
        for i in range(len(mutant)):
            if random.random() < self.mut_rate:
                mutant[i] = random.randint(0, 1)
        return mutant
                
    def onemax_fitness(self, indiv):
        fitness = sum(indiv[i] == self.target[i]
                      for i in range(0, len(self.target)))
        return fitness

    def evolve(self):
        parent = self.create_indiv()
        run = []
        #max_mut, min_mut = 0, 1
        #max_delt, min_delt = 0, 1
        success_mut = 0

        for _ in range(self.generations):
            child = self.mutate(parent)
            parent_fit = self.onemax_fitness(parent)
            child_fit = self.onemax_fitness(child)

            if child_fit > parent_fit:
                parent = child
                success_mut += 1
            run.append(parent_fit)

              # broken implementation of adaptive mutation
            if self.adaptive:
                if self.generations % self.eval_mod == 0:
                    success_rate = success_mut / self.eval_mod
                    print "suc_rate:", success_rate
                if success_rate > 0.2:
                    self.delta = self.delta * 2
                    self.mut_rate = gaussian(self.mut_rate, self.delta)
                else:
                    self.delta = self.delta / 0.2
                    self.mut_rate = gaussian(self.mut_rate, self.delta)
                    
                if self.mut_rate < 0:
                    print "wtf", self.mut_rate, self.delta
                    self.mut_rate = 0

          #     if self.delta > max_delt: max_delt = self.delta
          #     if self.delta < min_delt: min_delt = self.delta
          #     if self.mut_rate > max_mut: max_mut = self.mut_rate
          #     if self.mut_rate < min_mut: min_mut = self.mut_rate
          # print "delta max", max_delt, "min", min_delt
          # print "mutrate max", max_mut, "min", min_mut
        return run

def main():
    results = []
    evo = Evostrategy()
    
    for _ in range(10):
        run = evo.evolve()
        results.append(run)
        
    graph.plot_3d(evo.generations, results)
    graph.plot_ave(evo.generations, results)

if __name__ == "__main__":
    main()
