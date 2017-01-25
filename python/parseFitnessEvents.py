import os, sys, math

def ave(values):
    return float(sum(values)) / len(values)

def std(values, ave):
    return math.sqrt(float(sum((value - ave) ** 2 
                               for value in values))/ len(values))

def parse_result(filename):
    results_file = open(filename, 'r')

    fit_gain = []
    fit_loss = []
    neutrals = 0
    invalids = 0
    for line in results_file:
        line = line.rstrip()
        line = line.split(',')
        change, direction = float(line[0]), int(line[1])
        if direction  == 1:
            fit_gain.append(change)
        elif direction == -1:
            fit_loss.append(change)
        elif direction == 0:
            neutrals += 1
        elif direction == -2:
            invalids += 1

    ave_gain, ave_loss =round( ave(fit_gain), 2), round(ave(fit_loss),2)
    std_gain, std_loss = std(fit_gain, ave_gain), std(fit_loss, ave_loss)
    std_gain, std_loss = round(std_gain,2), round(std_loss,2)

    result = {"good":len(fit_gain), "bad":len(fit_loss), "neutral":neutrals,
              "invalids":invalids, "gain":sum(fit_gain),"loss":sum(fit_loss),
              "ave_gain":ave_gain, "ave_loss":ave_loss, "std_gain":std_gain,
              "std_loss":std_loss}
    results_file.close()
    return result

nodal = parse_result('nodalCodonFitness.dat')
struct= parse_result('structuralCodonFitness.dat')
integer= parse_result('intflipCodonFitness.dat')

print "Beneficial Mutations & ",nodal['good'],"&",struct['good'],"&",integer['good'],"\\\\" 
print "Inferior Mutations &",nodal['bad'],"&",struct['bad'],"&",integer['bad'],"\\\\" 
print "Neutral Mutations & ",nodal['neutral'],"&",struct['neutral'],"&",integer['neutral'],"\\\\" 
print "Invalids & ",nodal['invalids'],"&",struct['invalids'],"&",integer['invalids'],"\\\\" 
print "Fitness Gain & ",nodal['gain'],"&",struct['gain'],"&",integer['gain'],"\\\\" 
print "Fitness Loss & ",nodal['loss'],"&",struct['loss'],"&",integer['loss'],"\\\\" 
print "Average Gain & ",nodal['ave_gain'],"+-",nodal['std_gain'],"&",struct['ave_gain'],"+-",struct['std_gain'],"&",integer['ave_gain'],"+-",struct['std_gain'],"\\\\" 
print "Average Loss & ",nodal['ave_loss'],"+-",nodal['std_loss'],"&",struct['ave_loss'],"+-",struct['std_loss'],"&",integer['ave_loss'],"+-",struct['std_loss'],"\\\\" 

