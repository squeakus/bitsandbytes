"""This class contains demo methods for generating graphs. The methods
 use different functions from the geometry class to see if they look
 like pylons.  The results will be saved in pylon.mesh, which can then
 be viewed using medit. there is also commented out code at the end of
 the main function that will run the structural analysis, but this
 wont work until we make everything integer based and tidy up the apply
 stresses method in analyser.py"""
import graph, analyser, grammar, subprocess
from geometry import *
from constraints import equidistant_line_placement, optimal_line_placement
from math import sin, pi, sqrt, pow

def lt(val_a, val_b):
    """less than op for conditionals in generated code"""
    return operator.lt(val_a, val_b)


def le(val_a, val_b):
    """less than or equal op for conditionals in generated code"""
    return operator.le(val_a, val_b)


def gt(val_a, val_b):
    """greater than op for conditionals in generated code"""
    return operator.gt(val_a, val_b)


def lambda_graph():
    pylon_graph = graph.graph()
    centre_ids = []
    all_brace_ids = []
    sections = 5
    subsecs = [3, 5, 5, 5, 3]
    max_width = 5000
    min_width = 500
    max_height = 55000
    #origin, base, arm1, arm2, arm3, top
    height_sizes = [10, 5, 5, 5, 4/4]
    #this list is inverted as width decreases
    width_sizes = [10, 5, 5, 10, 40]
    brace_type = [2, 3, 3, 3, 3]
    height = scaled_list(0, max_height, height_sizes)
    width = scaled_list(0, max_width, width_sizes, True)



    #line_points = equidistant_line_placement(5000, True)
    line_points = optimal_line_placement(6)
    line_offset = (max_width * 1.5, 0 , height[2])
    line_points = line_points[0:3]
    
    #offset to correct height
    for i in range(len(line_points)):
        line_points[i] = pt_plus_pt(line_points[i],line_offset)

    print "linepts:", line_points
    height[3] = line_points[1][2] * 1.05
    height[4] = line_points[2][2] * 1.05


    def centre(t):
        pt_a, pt_b = section['base'], section['top']
        retval = (lambda t: interpolate(t, (pt_a, pt_b)))(t)
        return retval

    def brace(t):
        pt_a, pt_b = section['base_offset'], section['top_offset']
        retval = (lambda t: pt_plus_pt((lambda t: interpolate(t, (pt_a, pt_b)))(t),
                                       (lambda t: centre(t))(t)))(t)
        return retval

    def cross_brace(base_id, t):
        n = section['brace_type']
        current_ids = []
        for j in range(n):
            t_val = t + (j - (n - 1) / 2.0) / float(section['subsecs'])
            if t_val > 1 : t_val = 1
            elif t_val < 0: t_val = 0
            xyz = brace(t_val)
            if xyz[2] == 0:
                new_id = pylon_graph.add_unique_node(xyz, 'base')
            else:
                new_id = pylon_graph.add_unique_node(xyz, 'crossbrace')
            all_brace_ids.append(new_id)
            current_ids.append(new_id)
            pylon_graph.add_edge(base_id, new_id)
        return current_ids

    def create_arm(connectors, idx):
        print "createArm:", connectors
        startpoint = line_points[idx]
        top_point = pylon_graph.node[connectors[0]]['xyz']
        base_point = pylon_graph.node[connectors[1]]['xyz']
        triangle = [startpoint, top_point, base_point]

        line_id = pylon_graph.add_unique_node(startpoint, 'line')
        pylon_graph.varinski(triangle, "arm", [1,2,2,2])


    def make_section(level):
        brace_ids = []
        for i in range(section['subsecs'] + 1):
            t = i / float(section['subsecs'])
            node_id = pylon_graph.add_unique_node(centre(t), 'centre')
            centre_ids.append(node_id)
            brace_ids.extend(cross_brace(node_id, t))
                
        if level in range(1, 4):
            brace_set = list(set(brace_ids))
            brace_set.sort()
            connections = 2
            connectors = [brace_set[-1], brace_set[-3]]
            create_arm(connectors, level - 1)

    for x in range(sections):
        print "section:", x
        section = dict(base=[0, width[x], height[x]],
                       top=[0, width[x+1], height[x+1]],
                       base_offset=[width[x],0,0],
                       brace_type=brace_type[x],
                       top_offset=[width[x+1],0,0],
                       subsecs=subsecs[x])  
        make_section(x)

    #pylon_graph.connect_nodes(centre_ids)
    pylon_graph.connect_nodes(all_brace_ids)
    pylon_graph.node[all_brace_ids[-1]]['label'] = 'ground'
    rotated_graph = pylon_graph.copy_and_rotate_around_xy_plane(pylon_graph,180)
    mirror_graph = pylon_graph.copy_and_offset_with_mirror(rotated_graph, [0, 0, 0], True)
    full_graph = pylon_graph.copy_and_rotate_around_xy_plane(mirror_graph, 90)
    final_graph = pylon_graph.sanitise_pylon(full_graph, width[1])
    pylon_graph.replace_graph(final_graph)
    return pylon_graph

def main():
    """ Method for testing pylon creating functions."""
    GRAPH = lambda_graph()
    GRAPH.save_graph("pylon")
    meshName = "pylon.mesh"
    cmd = "./population/linuxShow "+meshName
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    process.communicate()
    print "nodes:", GRAPH.number_of_nodes()
    print "edges", GRAPH.number_of_edges()


#will it blend?
    azr = analyser.Analyser('test',"moo",True)
    azr.my_graph = GRAPH
    azr.assign_load_case()	
    azr.parse_graph()
    azr.apply_stresses()
    azr.create_slf_file()
    azr.test_slf_file()
    azr.parse_results()
    azr.show_analysis()

if __name__ == '__main__':
    main()
