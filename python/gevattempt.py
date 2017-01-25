#!/usr/bin/env python                                                                                   

import sys, os, re
homedir = os.environ['HOME']
sys.path.append(homedir+"/Documents/interactive3d")
sys.path.append(homedir+"/Documents/interactive3d/blender")

import pexpect
import sys

# The idea is that the blender python code regards this Geva object
# as the EC algorithm. It doesn't know what's happening inside it or 
# anything about the GEVA program etc. It just calls methods on the Geva
# object and gets strings back.

# These are the kinds of calls to be made from gui (or from a test-harness program):
# g = geva.Geva()
# inds = g.get_generation()
# g.set_fitness(adfjfakdf)

class Geva():
    GEVA_cmd = "java -jar GEVA.jar -properties_file ../param/Parameters/Experiments/ShapeGrammar/PictureCopy.properties -main_class Main.RunInteractive -population_size 5 -generations 10"
    GEVA_prompt = "GEVA CLI:"

    def __init__(self,GEVA_dir):
        # FIXED! how to set the path?        
        os.chdir(GEVA_dir)
        # FIXME what args, timeouts etc should we use? EOF?
        self.child = pexpect.spawn(Geva.GEVA_cmd)
        self.child.expect(".*" + Geva.GEVA_prompt)


    def get_generation(self):
        # tell geva to print the new generation
        # get the strings
        # wait for prompt
        # return them
        self.child.sendline("print")
        index = self.child.expect([(".*")+ Geva.GEVA_prompt, pexpect.EOF])
        if index == 0:
            population = self.child.match.group();
            return population
        else:
            print "something dodgy happened in get_generation()"
            os.exit

    # FIXME should we just tell GEVA to iterate, and then write another
    # method to get each individual one by one?
    def get_next_generation(self):
        # make sure all fitness values are set (or defaulted?)
        # send them to geva?
        # tell geva to iterate
        # wait for prompt
 
 # First attempt at parsing the string
 #       oldpop = str(self.get_generation())
 #       oldpop = oldpop.split('\n')
 #       expr = re.compile('fit')
 #       for individual in oldpop:
 #           print "hello!"+individual
 #           result = expr.search(individual)
 #           print result.span()
 #           print result.end()
 #           print result

        self.child.sendline("iterate")
        index = self.child.expect([".*"+ Geva.GEVA_prompt, pexpect.EOF])
        if index == 0:
            print "ITERATING!!!"
            iteration = self.child.match.group();
            print iteration
            return self.get_generation()
        else:
            print "something dodgy happened in get_next_generation()"
            os.exit



    def set_fitness(self,idx, fitness):
        # set a fitness value
        # wait for prompt
        pass

