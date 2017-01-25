"""
Evolveit reads the ranges set in the variable file, it will then
scan your program and set the variable value for the first instance it 
finds in the code. The fitness results has to be stored in a variable 
called fitnessresult.
"""

import random, sys, itertools

class Evostrategy:
    """A population based evolutionary strategy using the 1/5th rule
    for adaptive mutation. An inital parent population is created and 
    mutated to create a child population. Successive iterations 
    take the best of the parent and child populations and return a 
    mutated child population"""    
    def __init__(self, variables, pop_size):
        self.variables = variables
        self.pop_size = pop_size
        self.gen_count = 0
        self.mut_rate = 0.1
        self.success_mut = 0
        self.adaptive = False #change mutrate during run
        self.pop = []
        self.initialise_pop()

    def initialise_pop(self):
        """set codons to value in variable range"""
        for _ in range(self.pop_size):
            self.pop.append({'genome':self.create_indiv(),
                             'fitness':0})

    def create_indiv(self):
        """initialise a binary individual"""
        genome = []
        for var in self.variables:
            value = random.choice(var['vals'])
            genome.append(value)
        return genome

    def mutate(self, genome):
        """mutate each codon with a given probability"""
        mutant = list(genome)
        for i in range(len(mutant)):
            if random.random() < self.mut_rate:
                mutant[i] = random.choice(self.variables[i]['vals'])
        return mutant

    def adapt_mutation(self):
        """If more than 1/5 of the mutations are successful then 
        increase the mut rate, otherwise reduce it"""
        success_rate = float(self.success_mut) / self.pop_size
        # 1/5th rule
        if success_rate > 0.2:
            self.mut_rate = self.mut_rate * 2
        else:
            self.mut_rate = self.mut_rate / 2
        self.success_mut = 0
        # make sure self.mut_rate is within boundaries
        if self.mut_rate < 0.0005: 
            self.mut_rate = 0.0005
        if self.mut_rate > 0.5: 
            self.mut_rate = 0.5

    def iterate(self, children):
        """ Replace worst parents with best children.
        The bigger the number the better the fitness"""

        self.gen_count += 1
        self.pop.sort(key=lambda k: k['fitness'])
        children.sort(key=lambda k: k['fitness'])
        children.reverse()

        for child in children:
            for parent in self.pop:
                if parent['fitness'] < child['fitness']:
                    self.success_mut += 1      
                    parent['genome'] = child['genome']
                    parent['fitness'] = child['fitness']
                    break
                
        self.pop.sort(key=lambda k: k['fitness'])
        children = []
        for parent in self.pop:
            mutated_genome = self.mutate(parent['genome'])
            children.append({'genome':mutated_genome,
                             'fitness':0})
        return children

class Evaluator:
    """Parses the variables, reads the source, generates the 
    quines and then evaluates them."""
    
    def __init__(self, sourcename, variablename):
        self.variables = []
        self.source = []
        self.get_variables(variablename)
        self.read_quine(sourcename)
        self.quine_name = "quine.py"

    def get_variables(self, filename):    
        """parse the variable file and build a dictionary object""" 
        varfile = open(filename, 'r')
        for line in varfile:
            name = line.split('=')[0]
            name = name.replace(' ', '')
            self.variables.append({'name': name})
            exec(line)
        varfile.close()

        for var in self.variables:
            print var
            values, fitnessresult = 0, 0
            valexec = "values = " + var['name']
            exec(valexec)
            var['vals'] = values
            var['found'] = False

    def read_quine(self, sourcename):
        """Parse the source file into an array """
        srcfile = open(sourcename, 'r')
        for line in srcfile:
            self.source.append(line)
        
    def write_quine(self, genome):
        """Alter the variables and output a new source file"""
        newquine = []
        for line in self.source:
            for idx, var in enumerate(self.variables):
                if var['found'] == False:
                    if line.find(var['name']) > -1:
                        line = var['name'] + " = " + str(genome[idx]) + "\n" 
                        var['found'] = True
            newquine.append(line)        

        for var in self.variables:
            var['found'] = False

        outfile = open(self.quine_name, 'w')
        for line in newquine:
            outfile.write(line)
        outfile.close()
        
    def eval_quine(self):
        """Execute the quine and get the fitness value"""
        execfile(self.quine_name, globals())
        return fitnessresult

def main():
    """create an evaluator and an evostrategy and off you go!"""
    no_of_gens = 10000

    if(len(sys.argv) != 3):
        print "Usage: " + sys.argv[0] + " <programfile> <variablefile>"
        sys.exit(1)
    progstr = sys.argv[1]
    varstr = sys.argv[2]
    print "prog", progstr, "var",varstr   
    evaluator = Evaluator(progstr, varstr)
    evo = Evostrategy(evaluator.variables,50)
    
    # create an initial child population
    children = evo.iterate(evo.pop)

    for i in range(no_of_gens):
        # evaluate the children
        for child in children:
            evaluator.write_quine(child['genome'])
            child['fitness'] = evaluator.eval_quine()
            
        children = evo.iterate(children) # create next gen
        if evo.adaptive:
            evo.adapt_mutation()
        print "generation", i, " fitness:", evo.pop[-1]['fitness']

    print "best solution", evo.pop[-1], " fitness:", evo.pop[-1]['fitness'] 
            
if __name__ == "__main__":
    main()
