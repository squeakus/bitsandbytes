"""This class contains demo methods for generating graphs. The methods
 use different functions from the geometry class to see if they look
 like pylons.  The results will be saved in pylon.mesh, which can then
 be viewed using medit. there is also commented out code at the end of
 the main function that will run the structural analysis, but this
 wont work until we make everything integer based and tidy up the apply
 stresses method in analyser.py"""
import graph, analyser, grammar, subprocess
from geometry import *
from constraints import equidistant_line_placement, optimal_line_placement, line_configuration
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
    sections = 6
    subsecs = [1, 3, 5, 5, 5, 3]
    max_width = 5000
    min_width = 500
    max_height = 55000
    line_points = line_configuration(70,110,0,False)
    
    height_sizes = [2, 2, 3]
    #this list is inverted as width decreases
    width_sizes = [10, 5, 5, 10, 40, 40]
    arm_type = [(1,1,2,1), (2,2,2,1),(2,1,2,2),(1,2,2,1)]
    brace_type = [3, 3, 2, 2, 2, 2]
    height = scaled_list(0, line_points[0][2], height_sizes)
    width = scaled_list(0, max_width, width_sizes, True)
    print "height", height
    line_offset = (width[3] + 4800, 0 , 0)
    #offset to correct height
    for i in range(len(line_points)):
        line_points[i] = pt_plus_pt(line_points[i],line_offset)

    for i in range(1,len(line_points)):
        height.append(line_points[i][2] * 1.05)
    height.append(height[5] * 1.05)
    print "max height", height[6]
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
        pylon_graph.varinski(triangle, 'arm', arm_type.pop())

    def create_legs():
        a_xyz = brace(1)
        b_xyz = brace(0)
        triangle = [centre(1), brace(1), brace(0)]
        node_ids = pylon_graph.varinski(triangle, "brace", arm_type.pop())
        for node_id in node_ids:
            xyz = pylon_graph.node[node_id]['xyz']
            if check_point_on_xyline(a_xyz, b_xyz, xyz):
                pylon_graph.node[node_id]['label'] = 'leg'

    def make_section(level):
        print "making section", level
        brace_ids = []
        if level == 0:create_legs()
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
    print "mirror"
    mirror_graph = pylon_graph.copy_and_offset_with_mirror(rotated_graph, [0, 0, 0], True)
    print "copying"
    full_graph = pylon_graph.copy_and_rotate_around_xy_plane(mirror_graph, 90)
    print "sanitizing"
    final_graph = pylon_graph.sanitise_pylon(full_graph, width[1])
    print "replacing"
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
    # azr = analyser.Analyser('test',"moo",True)
    # azr.my_graph = GRAPH
    # azr.assign_load_case()	
    # azr.parse_graph()
    # azr.apply_stresses()
    # azr.create_slf_file()
    # azr.test_slf_file()
    # azr.parse_results()
    # azr.show_analysis()

if __name__ == '__main__':
    main()
