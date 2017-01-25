"""Implementation of NSGA-II in Python

This python module implements Individual class representing single resolution
of the problem resolved by the algorithm.

See <http://code.google.com/p/py-nsga-ii>
for additional details and package information.

------------------------------------------------------------------
Author:    Michal Fita <michal.fita(.)gmail.com>
Date:      2007-11-30
Version:   0.1.0
Copyright: (c) 2007 Michal Fita
License:   Licensed under the Apache License, Version 2.0 (the"License");
	   you may not use this file except in compliance with the License.
	   You may obtain a copy of the License at

	       http://www.apache.org/licenses/LICENSE-2.0

	   Unless required by applicable law or agreed to in
	   writing, software distributed under the License is
	   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
	   CONDITIONS OF ANY KIND, either express or implied.  See
	   the License for the specific language governing
	   permissions and limitations under the License.
------------------------------------------------------------------
"""

import math
from fpconst import *
from random import uniform, random, betavariate
from copy import *

from Problem import *

__all__ = ['Individual']

last_hash = 763625

"""
This class contains definition of single individual in population.
"""
class Individual():
    "Some properties"
    rank = PosInf
    distance = PosInf
    genes = []
    func_vals = []
    
    "Initialize completly new individual, all genes are randomly choosen."
    "Initialize empty child for crossover."
    def __init__(self, problem, empty = False):
        if problem is None or not isinstance(problem.obj_funcs, list) or len(problem.obj_funcs) < 2:
            raise TypeError('Problem passed to constructor has to have obj_funcs')
        
        "Store the problem to get functions list for later use."
        self.problem = problem
        
        self.genes = []
        if empty is False:
            "Prepare completely new individual"
            for g in range(1, problem.noVars + 1):
                v = uniform(problem.var_bounds[0], problem.var_bounds[1])
                self.genes.append(v)
            "Evaluate objective functions for new individuals"
            self.evaluate()
        else:
            "Empty individual"
            for g in range(1, problem.noVars + 1):
                v = PosInf
                self.genes.append(v)

        
    """
    Evaluate objective functions for new individuals
    """        
    def evaluate(self):
        self.func_vals = []
        for f in self.problem.obj_funcs:
            fv = f(self.problem, self)
            self.func_vals.append(fv)
        #print "func_vals",str(self.func_vals)

    """
    This method guarantee that any gene does not cross boundaries defined in problem
    """        
    def normalize(self):
        for i in range(0, len(self.genes)):
            if self.genes[i] > self.problem.var_bounds[1]:
                "When value of gene crosses upper boundary set it to that boundary value"
                self.genes[i] = self.problem.var_bounds[1]
            elif self.genes[i] < self.problem.var_bounds[0]:
                "When value of gene crosses lower boundary set it to that boundary value"
                self.genes[i] = self.problem.var_bounds[0]
        
    """
    This method realizes the crossover for new individual (SBX).
    Result of the method is a list with two childrens.
    """
    def crossover(self, second_parent):
        childs = [Individual(self.problem, True), Individual(self.problem, True)]
        "For each gene in individual"
        j = 0;
        for g in self.genes:
            # - VARIANT 1
            #if abs(g - second_parent.genes[j]) > 1.0e-14:
            #    beta = 1.0 + (2.0 * (g - self.problem.var_bounds[0]) / abs(second_parent.genes[j] - g));
            #    alpha = 2.0 - pow(beta, -(self.problem.eta_c + 1.0));
            #    bq = betavariate(alpha, beta)
            #else:
            #    bq = 1.0
            # - VARIANT 2
            u = random()
            if u <= 0.5:
                bq = pow((2.0 * u), (1.0 / (self.problem.eta_c + 1.0)))
            else:
                bq = 1.0 / pow(((2.0 * (1.0 - u))), (1.0 / (self.problem.eta_c + 1.0)))
                
            ch1v = 0.5 * (((1.0 + bq) * g) + (1.0 - bq) * second_parent.genes[j])
            childs[0].genes[j] = ch1v # add new gene for child 1
            ch2v = 0.5 * (((1.0 - bq) * g) + (1.0 + bq) * second_parent.genes[j])
            childs[1].genes[j] = ch2v # add new gene for child 2 
            #print "ch1v [",j,"] = ", ch1v, "1par_g", g , " 2par_g = ", second_parent.genes[j], " bq = ", bq 
            
            j += 1
            
        childs[0].normalize()
        childs[0].evaluate()
        childs[1].normalize()
        childs[1].evaluate()
        return childs
    
    """
    This method realizes the mutation for new individual (polynomial mutattion).
    Result of the method is the individual itself (was: 'is new individual') after mutation.
    """
    def mutation(self):
        child = self # was deepcopy(self)
        for i in range(0, len(child.genes)):
            u = random()
            g = child.genes[i]
            if u <= self.problem.p_m:
                r = random()
                delta = 0.0
                if r < 0.5:
                    delta = pow(2.0 * r, (1.0 / (self.problem.eta_m + 1.0))) - 1.0
                else:
                    delta = 1 - pow(2.0 * (1.0 - r), (1.0 / (self.problem.eta_m + 1.0)))
                child.genes[i] = g + delta * (self.problem.var_bounds[1] - self.problem.var_bounds[0])    
        child.normalize()
        child.evaluate()
        return child
    
    """
    Method checks whether the individual is dominated by another.
    This function minimises, returns true if self dominates other
    """
    def dominates(self, other):
        dominated = False
        won,lost = False, False
        for i in range(0, len(self.func_vals)):
            if self.func_vals[i] < other.func_vals[i]:
                won = True
            elif self.func_vals[i] > other.func_vals[i]:
                lost = True
        if won == True and lost == False:
            dominated = True
        return dominated
           
