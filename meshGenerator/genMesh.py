#!/usr/bin/python                                                               
import sys
#import graph
import geometry
import graph
from math import *

program2="def mutant():{def make_strut(i, t):{n=4{}id = g.get_unused_nodeid(){}xyz=walkway(t){}xyz[2]=p*xyz[2]+(1-p)*handrail(t)[2]{}g.add_node(id, xyz=xyz){}g.add_edge(i, id){}for j in range(n):{id2 = g.get_unused_nodeid(){}xyz=handrail(t+(j-(n-1)/2.0)/float(npts)){}g.add_node(id2, xyz=xyz){}handrail_node_ids.append(id2){}g.add_edge(id, id2){}}}def handrail(t):{return (lambda t: geometry.pt_plus_pt((lambda t: geometry.pt_plus_pt((lambda t: [0.0, 0.05 * (1.0 + cos(5*0.95 * 4 * pi * t)), 0.0])(t), (lambda t: [0.0, 0.0, 0.2 * (1.0 + cos(5*0.05 * 4 * pi * t))])(t)))(t), (lambda t: geometry.pt_plus_pt(walkway(t), (lambda t: [0, 0, 4])(t)))(t)))(t)}def walkway(t):{retval=(lambda t: geometry.pt_plus_pt((lambda t: geometry.interpolate(t, (pointA, pointB)))(t), (lambda t: [0.0, 0.0, 4 * 0.75 * sin(pi * t)])(t)))(t){}return(retval)}def f(t):{return 1.0 - pow(2 * t - 1.0, 2)}strut_multiple = 2{}npts = strut_multiple*5{}pointA = [0, 0, 0]{}pointB = [30, 0, 0]{}p = 0.95{}g = graph.graph(){}handrail_node_ids=[]{}walkway_node_ids=[]{}for i in range(npts+1):{t=i/float(npts){}id = g.get_unused_nodeid(){}g.add_node(id, xyz=walkway(t), rung=True){}walkway_node_ids.append(id){}if i % strut_multiple == 0:{make_strut(id, t){}}}walkway_node_ids.sort(){}for i in range(len(walkway_node_ids) - 1):{g.add_edge(walkway_node_ids[i], walkway_node_ids[i+1])}handrail_node_ids.sort(){}for i in range(len(handrail_node_ids) - 1):{g.add_edge(handrail_node_ids[i], handrail_node_ids[i+1])}g.copy_and_offset_with_mirror((0.0, 5.5+0.1*0.35, 0), True){}return g.create_mesh()}XXXeval_or_exec_outputXXX = mutant()"

program = "def mutant():{def make_strut(i, t):{n=2{}id=g.get_unused_nodeid(){}xyz=walkway(t){}xyz[2]=p*xyz[2]+(1-p)*handrail(t)[2]{}g.add_node(id,xyz=xyz){}g.add_edge(i,id){}for j in range(n):{id2=g.get_unused_nodeid(){}xyz=handrail(t+(j-(n-1)/2.0)/float(npts)){}g.add_node(id2,xyz=xyz){}handrail_node_ids.append(id2){}g.add_edge(id,id2){}}}def handrail(t):{return (lambda t:geometry.pt_plus_pt((lambda t:[0.0,0.0,0.9967697625964739*(1.0+cos(5*0.7666786856794164*4*pi*t))])(t),(lambda t:geometry.pt_plus_pt(walkway(t),(lambda t:[0,0,4])(t)))(t)))(t)}def walkway(t):{retval=(lambda t:geometry.pt_plus_pt((lambda t:interpolate(t,(pointA,pointB)))(t),(lambda t:[0.0,0.0,4*0.26531205804334584*sin(pi*t)])(t)))(t){}return(retval)}def f(t):{return 1.0 - pow(2 * t - 1.0, 2)}strut_multiple=3{}npts=strut_multiple*5{}pointA = [0,0,0]{}pointB = [30,0,0]{}p=0.18317794575503932{}g=graph(){}handrail_node_ids=[]{}walkway_node_ids=[]{}for i in range(npts+1):{t=i/float(npts){}id=g.get_unused_nodeid(){}g.add_node(id,xyz=walkway(t),rung=True){}walkway_node_ids.append(id){}if i % strut_multiple == 0:{make_strut(id,t){}}}walkway_node_ids.sort(){}for i in range(len(walkway_node_ids) - 1):{g.add_edge(walkway_node_ids[i],walkway_node_ids[i+1])}handrail_node_ids.sort(){}for i in range(len(handrail_node_ids) - 1):{g.add_edge(handrail_node_ids[i],handrail_node_ids[i+1])}g.copy_and_offset_with_mirror((0.0,5.5+0.1*0.47378327207350324,0),True){}return g.create_mesh()}XXXeval_or_exec_outputXXX=mutant()"

# An unpleasant limitation in Python is the distinction between
# eval and exec. The former can only be used to return the value
# of a simple expression (not a statement) and the latter does not
# return anything.
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
    print retval
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

parsedProgram = python_filter(program)
print parsedProgram
eval_or_exec(program2)
num = 0.05
string =  ""
