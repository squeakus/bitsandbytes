"""Implementation of NSGA-II in Python

This python module implements Problem base class which is then extended
by specific problem. Classes for ZDT1 and ZDT2 are present in this file, what
is object to future changes.

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

from math import *
from random import choice
from copy import *
from fpconst import *

from Individual import *

__all__ = ['Problem', 'ProblemZDT1', 'ProblemZDT2']

"""
This class defines the problem in all details needed to run the experiment.
"""
class Problem(object):
    "Some important values"
    popSize = 100 # Size of population
    noVars = 30 # Number of variables
    noFuncs = 2  # Number of objective functions
    generations = 10 # Number of generations in experiment
    eta_c = 20 # Crossover distribution
    eta_m = 20 # Mutation distribution
    p_c = 0.9 # Crossover propability
    p_m = 0.1 # Mutation propability
    var_bounds = [0.0, 4.0]
    obj_funcs = []
    
    "Variables for experiment"
    pop = [] # Population
    
    "Initialize the problem for experiment"
    def __init__(self):
        "Create begining population"
        for i in range(1, self.popSize):
            self.pop.append(Individual(self))
         
    def fast_nondominated_sort(self, pop):
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
            fSize = len(fronts[i+1])
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
        print "No. of non-empty fronts:",i
        return fronts
                
    def crowding_distance_assignment(self, pop):
        popSize = len(pop)
        for indiv in pop:
            "Initialize distance for each individual"
            indiv.distance = 0
        
        #build a cumulative distance for each individual
        for m in range(0, self.noFuncs):
            "Sort using objective m"
            pop.sort(lambda x,y: cmp(x.func_vals[m], y.func_vals[m]))
            "Boundary points are always selected"
            pop[0].distance = PosInf
            pop[popSize-1].distance = PosInf
            for i in range(2, popSize-1):
                pop[i].distance += (pop[i+1].func_vals[m]-pop[i-1].func_vals[m])
        pop.sort(Problem.crowded_comparasion_operator)
    
    @staticmethod            
    def crowded_comparasion_operator(x,y):
        "Because we sort in descending order the return values are inverted"
        if x.rank < y.rank:
            return +1
        elif (x.rank == y.rank):
            #print "X:",x.distance," Y:",y.distance
            if x.distance > y.distance:
                return -1
            else:
                return +1
                
    """
    This method runs the whole experiment
    """
    def experiment(self):
        for iteration in range(0, self.generations):
            print "iteration =", iteration
            "Create child population newPop."
            self.newPop = []
            while len(self.newPop) < self.popSize:
                p1 = choice(self.pop)
                p2 = choice(self.pop)
                while hash(p1) == hash(p2):
                    p2 = choice(self.pop)
                "Append new pair of childrens crossed over from two parents to the list of childrens."
                self.newPop.extend(p1.crossover(p2))
            "Mutate all childrens"
            for c in self.newPop:
                c.mutation()        
            "Join both into one list"
            totalPop = []
            totalPop.extend(self.pop)
            totalPop.extend(self.newPop)
            "Do fast non-dominated sort on whole"
            fronts = []
            fronts = self.fast_nondominated_sort(totalPop)
            "Create parent population for next iteration"
            self.pop = []
            i = 0 #front counter
            #assign distance and append fronts to current population
            while len(self.pop) + len(fronts[i]) <= self.popSize:
                self.crowding_distance_assignment(fronts[i])
                self.pop.extend(fronts[i])
                i += 1

            "Sort in descending order using crowded comparasion operator"
            self.crowding_distance_assignment(fronts[i])
            fronts[i].sort(Problem.crowded_comparasion_operator)
            print "appending"
            for indiv in fronts[i][0 : self.popSize-len(self.pop)]:
                print "indiv: rank: ",indiv.rank, " dist: ",indiv.distance
                
            self.pop.extend(fronts[i][0 : self.popSize-len(self.pop)])
           # print "Total pop"
           # for idx,indiv in enumerate(self.pop):
           #     print "indiv:",idx," rank: ",indiv.rank, " dist: ",indiv.distance
        "Show non-dominated frontss"
        print len(fronts)
        for i,f in enumerate(fronts):
            if len(f) > 0:
                print "new front:",i," size ",len(f)
        "Put fronts into files"
        num = 0
        for f in fronts:
            if len(f) > 0:               
                file = open("Front" + repr(num).zfill(3) + ".dat", "w")
                for g in f:
                    t = repr(g.func_vals[0]).rjust(5) + " " + repr(g.func_vals[1]).rjust(5) + "\n"
                    file.write(t)
                file.close()
            num += 1
            
class ProblemZDT1(Problem):
    "Objective function 1 for ZDT1: f1(x) = x1"
    def f1(self, x):
        return x.genes[0]
    #f1 = classmethod(f1)
    
    "g(x) = 1 + 9 (sum(for i=2 to n : xi))/(n - 1)"
    def g(self, x):
        sx = 0.0;
        for i in range(1, self.noVars): # n genes numerated from 0
            sx += x.genes[i]
        r = 1.0 + 9.0 * (sx / (self.noVars - 1.0))
        return r
    
    "Objective function 2 for ZDT1: f2(x) = g(x) [1 - sqrt(x1/g(x))]"
    def f2(self, x):
        gx = self.g(x)
        #print "x =", x, "gx = ", gx, "x.genes[1] = ", x.genes[1]
        r = gx * (1.0 - sqrt(x.genes[0] / gx))
        return r
    
    obj_funcs = [f1, f2]
    
class ProblemZDT2(ProblemZDT1):
    "Objective function 1 for ZDT2: f1(x) = x1"
    def f1(self, x):
        return x.genes[0]
    
    "Objective function 2 for ZDT2: f2(x) = g(x) [1 - pow((x1 / g(x)), 2)]"
    def f2(self, x):
        gx = self.g(x)
        r = gx * (1.0 - pow((x.genes[0] / gx), 2.0))
        return r
    
    obj_funcs = [f1, f2]    
    
if __name__ == "__main__":
    p1 = ProblemZDT1()
    p2 = ProblemZDT2()
    
    p1.experiment()
    #p2.experiment()
    
