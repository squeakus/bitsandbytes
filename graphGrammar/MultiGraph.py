import analyser, subprocess, graph, os
from geometry import *

def eval_or_exec(s):
    s = python_filter(s)
    try:
        retval = eval(s)
    except SyntaxError:
        # SyntaxError will be thrown by eval() if s is compound,
        # ie not a simple expression, eg if it contains function
        # definitions, multiple lines, etc. Then we must use
        # exec(). Then we assume that s will define a variable
        # called "XXXeval_or_exec_outputXXX", and we'll use that.
        exec(s)
        retval = XXXeval_or_exec_outputXXX
    return retval

# Create correct python syntax. We use {} to track indentation,
# which is not ideal because of the clash with dict literals.
def python_filter(txt):
    counter = 0
    for char in txt:
        if char == "{":
            counter += 1
        elif char == "}":
            counter -= 1
        tabstr = "\n" + "  " * counter
        if char == "{" or char == "}":
            txt = txt.replace(char, tabstr, 1)
    txt = "\n".join([line for line in txt.split("\n") 
                     if line.strip() != ""])
    return txt

def read_graph(fileName):
    loadFile = open(fileName,'r')
    for line in loadFile:
      if line.startswith("phenotype:"):
        line = line.lstrip("phenotype:")
        phenotype = line
        g = eval_or_exec(phenotype)
        return g

def compare(graphA,graphB):
    if graphA.size() > graphB.size():
       bigger,smaller = graphA,graphB
    else:
        bigger,smaller = graphB,graphA
        
    totalDistance = 0
    for nodeId in bigger.node:
        pos = bigger.node[nodeId]['xyz']
        distance = smaller.nearest_node(pos[0],pos[1],pos[2])
        totalDistance  += distance
    totalDistance = round(totalDistance,3)
    return totalDistance

def parse_results(resultsFolder,target,header,opFile):
    idxCounter = 0 
    maxDist = 0
    minDist = 10000
    worst = None
    best = None
    resultFiles = []
    for fileName in os.listdir(resultsFolder):
        if fileName.startswith(header) and fileName.endswith('.dat'):
            resultFiles.append(fileName)
    output = open(opFile,'a')
    for fileName in resultFiles:
        idxCounter += 1
        fileName = resultsFolder+fileName
        results = open(fileName,'r')
        lines = iter(results)
        for line in lines:
            if line.startswith('time:'):
                currentTime = float(line.lstrip('time:').rstrip())
                lines.next()
                lines.next()
                phenotype = lines.next().lstrip("phenotype:")
                g = eval_or_exec(phenotype)
                distance = compare(g,target)
                if distance > maxDist:
                    maxDist = distance
                    worst = phenotype
                if distance < minDist:
                    minDist = distance
                    best = phenotype
                output.write(str(idxCounter)+" "+str(currentTime)+" "+str(minDist)+"\n")
    output.close()
    print "max",maxDist,"min",minDist
resultsFolder ="/home/jonathan/Jonathan/programs/IECresults/all/"
target1 = read_graph("./target1.dat")
target2 = read_graph("./target2.dat")
target3 = read_graph("./target3.dat")

print "target1_intf"
parse_results(resultsFolder,target1,"target1_intf","./intTarget1Euc.dat")
print "target1_both"
parse_results(resultsFolder,target1,"target1_both","./bothTarget1Euc.dat")
print "target2_intf"
parse_results(resultsFolder,target2,"target2_intf","./intTarget2Euc.dat")
print "target2_both"
parse_results(resultsFolder,target2,"target2_both","./bothTarget2Euc.dat")
print "target3_intf"
parse_results(resultsFolder,target3,"target3_intf","./intTarget3Euc.dat")
print "target3_both"
parse_results(resultsFolder,target3,"target3_both","./bothTarget3Euc.dat")
