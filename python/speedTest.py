#! /usr/bin/env python

#Speed Testing 

import math
import os
import time
import subprocess

def speedTest():
    lua_args = {}
    LUA_DIR = '/home/jonathan/Jonathan/programs/lua'
    lua = {'dir':LUA_DIR,
                'cmd':['/usr/local/bin/lua  '+LUA_DIR + '/luGE.lua'],
                'args':lua_args}

    PROCESSORS = 2
    GEVA_DIR = '/home/jonathan/Downloads/GEVA-v1.2'
    geva_args ={'-properties_file':GEVA_DIR + '/param/Parameters/Simple_Letter.properties', 
                '-main_class':'Main.Run',
                '-grammar_file':GEVA_DIR + '/param/Grammar/simple_letter.bnf'} 
    geva = {'dir':GEVA_DIR,
            'cmd':['java', '-server', '-Xmx512m', '-jar',  GEVA_DIR + '/bin/GEVA.jar'],
            'args':geva_args}

#    experiments = {'lua':lua,'GEVA':geva}
    experiments = {'GEVA':geva}

    runs = 10
    for name, experiment in experiments.iteritems():
        tot_times = []
        fitnesses = []
        for run in range(0, runs):
            directory = os.getcwd() + '/' + name + '/'
            if not os.path.exists(directory):
                os.mkdir(directory)
            current_args = []
            for key, value in experiment['args'].iteritems():
                current_args.append(key)
                current_args.append(value) 
            current_cmd = experiment['cmd'] + current_args
            print(current_cmd)
            start = time.time()
            p = subprocess.Popen(current_cmd, bufsize=1024, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            #Nessecary to wait?
            p.wait()  
            tot_times.append((time.time() - start))
            for line in p.stdout:
                if name is lua and line.startswith('\"best'):
                    #Trimming "\n
                    fitnesses.append(float(line.split(":")[1][:-2]))
                elif name is 'GEVA' and line.startswith('Rank'):
                    print line
                    cols = line.split(" ")
                    fitnesses.append(float(cols[1].split(":")[1]))
        print("Fitness:",runs)
        statistics(fitnesses, name)
        print("Speed(ms)")
        statistics(tot_times, name)

def statistics(times, name):
    tot = sum(times)
    print "total",str(tot),"length",str(len(times))
    ave = tot/len(times)
    times.sort()
    max_time = times[-1]
    min_time = times[0]
    median = times[len(times)/2]
    var = 0
    for time in times:
        var =+ (time - ave)**2
    var = var / len(times)
    std = math.sqrt(var)
    
    print("%s: ave:%.3f std:%.3f med:%.3f min:%.3f max:%.3f" % (name, ave, std, median, min_time, max_time)) 

if __name__=='__main__':
    speedTest()
