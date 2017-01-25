import analyser, subprocess, graph 
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

def compare(bigger,smaller):
    totalDistance = 0
    for nodeId in bigger.node:
        pos = bigger.node[nodeId]['xyz']
        distance = smaller.nearest_node(pos[0],pos[1],pos[2])
        totalDistance  += distance
    totalDistance = round(totalDistance,3)
    print "totalDistance:",totalDistance
        

fileName1 = "./target3.dat"
fileName2 ="./target3.dat"

g1 = read_graph(fileName1)
g2 = read_graph(fileName2)
print "g1:",g1.size()
print "g2:",g2.size()

if g1.size() > g2.size():
    compare(g1, g2)
else:
    compare(g2, g1)
