import analyser, subprocess, graph, os
from geometry import *
from math import exp

def mutant():
  print "calling mutant"
  g = graph.graph()
  subdivide_line([100,0,0],[1000,900,900],2)
  
  handrail_node_ids=[]
  arm_node_ids=[]
  support_node_ids=[]
  brace_node_ids=[]
  strut_multiple = 1
  arm_multiple = 4
  npts = 3
  width = 1000
  height = 5000
  pointA = [0, 0, 0]
  pointB = [0, 0, height]

  def brace(t):
    retval = (lambda t: pt_plus_pt((lambda t: [width, 0.0, 0.0])(t), 
                                   (lambda t: pt_plus_pt(support(t),[0,0,0]))(t)))(t)
    return retval

  def arm(t):
    retval = (lambda t: pt_plus_pt((lambda t: [width * 3, 0.0, 0.0])(t), 
                                   (lambda t: pt_plus_pt(support(t),[0,0,0]))(t)))(t)
    return retval


  def support(t):
    retval=(lambda t: pt_plus_pt((lambda t: interpolate(t, (pointA, pointB)))(t), 
                                 (lambda t: [0.0, 0.0, 0.0])(t)))(t)
    return retval


  def brace_strut(base_id, t):
    n=2
    current_ids = []
    #xyz=support(t)
    #node_id=g.add_unique_node(xyz,'post')
    #g.add_edge(base_id, node_id)
    for j in range(n):
      t_val = t+(j-(n-1)/2.0)/float(npts)
      if not t_val > 1 and not t_val < 0:
        xyz=brace(t_val)
        id2=g.add_unique_node(xyz,'brace')
        brace_node_ids.append(id2)
        current_ids.append(id2)
        g.add_edge(base_id, id2)
        g.save_graph()
    return current_ids

  def arm_strut(brace_ids, t):
    print "starting arm:", brace_ids
    arm_ids = []
    startpoint = arm(t)
    g.add_unique_node(startpoint, 'line')
    for brace_id in brace_ids:
      endpoint = g.node[brace_id]['xyz']
      dist = distance(startpoint, endpoint)
      arm_points = subdivide_line(startpoint, endpoint, 3)
      print "calling arm add nodes"
      arm_ids.append(g.add_nodes(arm_points, "arm", True)) 
    print "arm ids:",arm_ids
    g.weave_nodes(arm_ids[0], arm_ids[1],1)
    #g.connect_neighbours(arm_ids, 1800, True)


#########
  for i in range(npts+1):
    "keep t as a fraction"
    t=i/float(npts)
    node_id=g.add_unique_node(support(t),'support')
    support_node_ids.append(node_id)
    brace_nodes = brace_strut(node_id, t)
    if i % arm_multiple == 1:
      arm_strut(brace_nodes, t)

  g.connect_nodes(brace_node_ids)
  g.connect_nodes(support_node_ids)
  #g.copy_and_offset_with_mirror((0.0,width, 0), True)


  return g

#XXXeval_or_exec_outputXXX = mutant()
testGraph = mutant()
testGraph.save_graph("test")

analyser = analyser.Analyser('test',"moo",True)
analyser.my_graph=testGraph
analyser.parse_graph()


#using medit to show the graph
meshName = "test.mesh"
cmd = "./population/linuxShow "+meshName
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
process.communicate()

#using slffea to show the mesh
#analyser.apply_stresses()
#analyser.create_slf_file()
#analyser.test_slf_file()
#analyser.parse_results()
#analyser.print_stresses()
#analyser.show_analysis()

#using matplot to show the graph
#analyser.myGraph.show_picture()
