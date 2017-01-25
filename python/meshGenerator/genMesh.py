#!/usr/bin/python                                                               
import sys
#import graph
import geometry
from math import *
program2= "def mutant():{def make_strut(i, t):{n=2{}id=g.get_unused_nodeid(){}xyz=walkway(t){}xyz[2]=p*xyz[2]+(1-p)*handrail(t)[2]{}g.add_node(id,xyz=xyz){}g.add_edge(i,id){}for j in range(n):{id2=g.get_unused_nodeid(){}xyz=handrail(t+(j-(n-1)/2.0)/float(npts)){}g.add_node(id2,xyz=xyz){}handrail_node_ids.append(id2){}g.add_edge(id,id2){}}}def handrail(t):{return (lambda t:geometry.pt_plus_pt((lambda t:[0.14626214008138616*(1.0+cos(5*0.6022194915461445*4*pi*t)),0.0,0.0])(t),(lambda t:geometry.pt_plus_pt(walkway(t),(lambda t:[0,0,4])(t)))(t)))(t)}def walkway(t):{retval=(lambda t:geometry.pt_plus_pt((lambda t:interpolate(t,(pointA,pointB)))(t),(lambda t:[0.0,0.0,4*0.9791225464917359*sin(pi*t)])(t)))(t){}return(retval)}def f(t):{return 1.0 - pow(2 * t - 1.0, 2)}strut_multiple=3{}npts=strut_multiple*10{}pointA = [0,0,0]{}pointB = [30,0,0]{}p=0.531480719117206{}g=geometry.graph(){}handrail_node_ids=[]{}walkway_node_ids=[]{}for i in range(npts+1):{t=i/float(npts){}id=g.get_unused_nodeid(){}g.add_node(id,xyz=walkway(t),rung=True){}walkway_node_ids.append(id){}if i % strut_multiple == 0:{make_strut(id,t){}}}walkway_node_ids.sort(){}for i in range(len(walkway_node_ids) - 1):{g.add_edge(walkway_node_ids[i],walkway_node_ids[i+1])}handrail_node_ids.sort(){}for i in range(len(handrail_node_ids) - 1):{g.add_edge(handrail_node_ids[i],handrail_node_ids[i+1])}g.copy_and_offset_with_mirror((0.0,5.5+0.1*0.15368599777747224,0),True){}return g.create_mesh()}XXXeval_or_exec_outputXXX=mutant()"

program = "def mutant():{def make_strut(i, t):{n=2{}id=g.get_unused_nodeid(){}xyz=walkway(t){}xyz[2]=p*xyz[2]+(1-p)*handrail(t)[2]{}g.add_node(id,xyz=xyz){}g.add_edge(i,id){}for j in range(n):{id2=g.get_unused_nodeid(){}xyz=handrail(t+(j-(n-1)/2.0)/float(npts)){}g.add_node(id2,xyz=xyz){}handrail_node_ids.append(id2){}g.add_edge(id,id2){}}}def handrail(t):{return (lambda t:geometry.pt_plus_pt((lambda t:[0.0,0.0,0.9967697625964739*(1.0+cos(5*0.7666786856794164*4*pi*t))])(t),(lambda t:geometry.pt_plus_pt(walkway(t),(lambda t:[0,0,4])(t)))(t)))(t)}def walkway(t):{retval=(lambda t:geometry.pt_plus_pt((lambda t:interpolate(t,(pointA,pointB)))(t),(lambda t:[0.0,0.0,4*0.26531205804334584*sin(pi*t)])(t)))(t){}return(retval)}def f(t):{return 1.0 - pow(2 * t - 1.0, 2)}strut_multiple=3{}npts=strut_multiple*5{}pointA = [0,0,0]{}pointB = [30,0,0]{}p=0.18317794575503932{}g=geometry.graph(){}handrail_node_ids=[]{}walkway_node_ids=[]{}for i in range(npts+1):{t=i/float(npts){}id=g.get_unused_nodeid(){}g.add_node(id,xyz=walkway(t),rung=True){}walkway_node_ids.append(id){}if i % strut_multiple == 0:{make_strut(id,t){}}}walkway_node_ids.sort(){}for i in range(len(walkway_node_ids) - 1):{g.add_edge(walkway_node_ids[i],walkway_node_ids[i+1])}handrail_node_ids.sort(){}for i in range(len(handrail_node_ids) - 1):{g.add_edge(handrail_node_ids[i],handrail_node_ids[i+1])}g.copy_and_offset_with_mirror((0.0,5.5+0.1*0.47378327207350324,0),True){}return g.create_mesh()}XXXeval_or_exec_outputXXX=mutant()"


def interpolate(p, xy):
        p = 1 - p
        x, y = xy
        x0, y0, z0 = x
        x1, y1, z1 = y
        return [x0 * p + x1 * (1 - p), y0 * p + y1 * (1 - p), z0 * p + z1 * (1 - p)]
# perform a single evaluation
def eval_indiv(ind):
    if ind['phen'] != "":
        ind['objects'] = self.eval_or_exec(ind['phen'])
    else:
        self.status_msg("Mutation broke the individual")
        ind['objects'] = []



def display_individual(self,individual):
    render.drawShapesEval(individual['objects'])



# An unpleasant limitation in Python is the distinction between
# eval and exec. The former can only be used to return the value
# of a simple expression (not a statement) and the latter does not
# return anything.
def eval_or_exec(s):
    s = python_filter(s)
    #print s
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
