import analyser, subprocess, graph, os
import operator
from geometry import *
from math import exp
from constraints import *

def lt(val_a, val_b):
    """less than op for conditionals in generated code"""
    return operator.lt(val_a, val_b)


def le(val_a, val_b):
    """less than or equal op for conditionals in generated code"""
    return operator.le(val_a, val_b)


def gt(val_a, val_b):
    """greater than op for conditionals in generated code"""
    return operator.gt(val_a, val_b)


def mutant():
  pylon_graph = graph.graph()
  centre_ids = []
  all_brace_ids = []
  max_width = 5000
  max_height = 55000
  sections=6
  subsecs = [1, 4, 4, 3, 5, 3]
  brace_type = [2, 3, 2, 2, 2, 3]
  height_sizes = [7, 9, 9, 7, 6,6 / 2]
  width_sizes = [10+1, 2, 2, 2, 4*5,1*5]
  height = scaled_list(0, max_height, height_sizes)
  width = scaled_list(0, max_width, width_sizes, True)
  arm_type = [(2,2,1,2), (2,1,2,1),(2,1,2,2),(2,1,1,2)]
  line_points = line_configuration(80, 0)
  line_offset = (max_width *1.1, 0 , 0)
  for i in range(len(line_points)):
    line_points[i] = pt_plus_pt(line_points[i],line_offset)
  height[3]=line_points[0][2]
  height[4] = line_points[1][2] * 1.03
  height[5] = line_points[2][2] * 1.05
  height[6] = height[5] * 1.03
  def centre(t):
    pt_a, pt_b = section['base'], section['top']
    retval = (lambda t: interpolate(t, (pt_a, pt_b)))(t)
    return retval
  def brace(t):
    pt_a, pt_b = section['base_offset'], section['top_offset']
    retval = (lambda t: pt_plus_pt((lambda t: interpolate(t, (pt_a, pt_b)))(t),(lambda t: centre(t))(t)))(t)
    return retval
  def cross_brace(base_id, t):
    n = section['brace_type']
    current_ids = []
    for j in range(n):
      t_val = t + (j - (n - 1) / 2.0) / float(section['subsecs'])
      if gt(t_val, 1): t_val = 1
      elif lt(t_val,0): t_val = 0
      xyz = brace(t_val)
      new_id = pylon_graph.add_unique_node(xyz, 'crossbrace')
      all_brace_ids.append(new_id)
      current_ids.append(new_id)
      pylon_graph.add_edge(base_id, new_id)
    return current_ids
  def create_arm(connectors, idx):
    startpoint = line_points[idx]
    top_point = pylon_graph.node[connectors[0]]['xyz']
    base_point = pylon_graph.node[connectors[1]]['xyz']
    triangle = [startpoint, top_point, base_point]
    line_id = pylon_graph.add_unique_node(startpoint,'line')
    pylon_graph.varinski(triangle, 'arm', arm_type.pop())
  def make_section(level):
    brace_ids = []
    if level == 0:
      triangle = [centre(1), brace(1), brace(0)]
      pylon_graph.varinski(triangle, "leg", arm_type.pop())
    else:
      for i in range(section['subsecs'] + 1):
        t = i / float(section['subsecs'])
        node_id = pylon_graph.add_unique_node(centre(t), 'centre')
        centre_ids.append(node_id)
        brace_ids.extend(cross_brace(node_id, t))
    if level in range(2, 5):
      brace_set = list(set(brace_ids))
      brace_set.sort()
      connections = 2
      connectors = [brace_set[-1], brace_set[-3]]
      create_arm(connectors, level - 2)
  for x in range(sections):
    section = dict(base=[0, width[x], height[x]],top=[0, width[x+1], height[x+1]],base_offset=[width[x],0,0],brace_type=brace_type[x],top_offset=[width[x+1],0,0],subsecs=subsecs[x])
    make_section(x)
  pylon_graph.connect_nodes(all_brace_ids)
  pylon_graph.node[all_brace_ids[-1]]['label'] = 'ground'
  rotated_graph = pylon_graph.copy_and_rotate_around_xy_plane(pylon_graph,180)
  mirror_graph = pylon_graph.copy_and_offset_with_mirror(rotated_graph, [0, 0, 0], True)
  full_graph = pylon_graph.copy_and_rotate_around_xy_plane(mirror_graph, 90)
  final_graph = pylon_graph.sanitise_pylon(full_graph, width[1])
  pylon_graph.replace_graph(final_graph)
  return pylon_graph



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
