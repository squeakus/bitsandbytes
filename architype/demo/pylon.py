"""This class contains demo methods for generating graphs. The methods
 use different functions from the geometry class to see if they look
 like pylons.  The results will be saved in pylon.mesh, which can then
 be viewed using medit. there is also commented out code at the end of
 the main function that will run the structural analysis, but this
 wont work we make everything integer based and tidy up the apply
 stresses method in analyser.py"""
import graph, analyser, grammar, subprocess
from geometry import *
from math import sin, pi, sqrt, pow

LENGTH = 1000 #make it big to keep it in integers
WIDTH = 5000 
HEIGHT = 55000
ORIGIN = [0, 0, 0]
LEVELS = 3


def star_graph():
    """Playing with the star function"""
    pylon_graph = graph.graph()
    idx = pylon_graph.add_unique_node(ORIGIN, "base")
    star_list = pylon_graph.add_star_to_node(idx, 6)
    pylon_graph.connect_nodes(star_list)
    pylon_graph.save_graph("star")
    return pylon_graph


def offset_graph():
    """build pylon using offset function"""
    pylon_graph = graph.graph()
    base = square(ORIGIN, LENGTH)
    base_ids = pylon_graph.add_nodes(base, "base")
    pylon_graph.connect_neighbours(base_ids, LENGTH)
    all_ids = []
    for i in range(LEVELS):
        level = offset(base, LENGTH * i, "z")
        level_ids = pylon_graph.add_nodes(level, "level" + str(i))
        all_ids.extend(level_ids)
    pylon_graph.connect_neighbours(all_ids, LENGTH)
    return pylon_graph


def rectangle_graph():
    """build pylon using offset function"""
    scaled = scale((200, 200, 200), 2)
    print scaled
    pylon_graph = graph.graph()
    base = rectangle(ORIGIN, WIDTH, LENGTH)
    base_ids = pylon_graph.add_nodes(base, "base")
    pylon_graph.connect_neighbours(base_ids, LENGTH)
    pylon_graph.connect_neighbours(base_ids, WIDTH)
    all_ids = []
    for i in range(LEVELS):
        level = offset(base, LENGTH * i, "z")
        level_ids = pylon_graph.add_nodes(level, "level" + str(i))
        all_ids.extend(level_ids)
    pylon_graph.connect_neighbours(all_ids, LENGTH)
    pylon_graph.connect_neighbours(all_ids, WIDTH)
    return pylon_graph


def sinusoid_graph():
    "generate sine graph"
    pylon_graph = graph.graph()
    points = sinusoid(ORIGIN, [LENGTH,0,0], WIDTH, LENGTH)
    print "points"
    ids = pylon_graph.add_nodes(points, "curve", True)
    return pylon_graph


def bezier_graph():
    """trying to generate a nice bezier curve"""
    pylon_graph = graph.graph()
    bezier_points = []
    for x in range(25):
        bezier_points.append(bezier_form(x * 0.01, ([4*0.1, 4*0.6, 4*0.45],
                                                  [4*1.0, 4*0.2, 4*0.8],
                                                  [4*0.8, 4*0.95, 4*0.95],
                                                  [4*0.55, 4*0.75, 4*0.8])))
    pylon_graph.add_nodes(bezier_points, "curve", True)
    mirror_points = mirror(bezier_points, 'x')
    pylon_graph.add_nodes(mirror_points, "curve", True)
    return pylon_graph


def rotated_graph():
    """creates a curve and then rotates it around xy plane"""
    pylon_graph = graph.graph()
    straight_points = []
    curve_points = []

    for x in range(10):
        current_point = interpolate(x * 0.1, (ORIGIN, [0, 0, LENGTH*3]))
        straight_points.append(current_point)
        curve_point = pt_plus_pt(current_point,
                                 [0.0,
                                  LENGTH * sin(pi * x * 0.1),
                                  0.0])
        curve_points.append(curve_point)
        #print "straight", current_point, "curve:", curve_point
    pylon_graph.add_nodes(straight_points, "straight", True)
    pylon_graph.add_nodes(curve_points, "curve", True)

    pylon_graph.rotate_points_around_xy_plane(curve_points, 6)
    #pylon_graph.connect_all_neighbours(LENGTH)
    pylon_graph.connect_by_height()
    return pylon_graph

def my_pylon_graph():
    x_width = sqrt(pow(WIDTH, 2) / 2)
    print "xwidth:", x_width
    points = [(WIDTH, 0, 0)]
    points.append((x_width * .4, 0, HEIGHT * 0.5))
    points.append((x_width * .4, 0, HEIGHT * 0.9))
    points.append((0, 0, HEIGHT))
    pylon_graph = graph.graph()
    pylon_graph.add_nodes(points, "start", True)
    pylon_graph.rotate_points_around_xy_plane(points, 4)
    pylon_graph.connect_by_offset_height()
    return pylon_graph


def lambda_graph():
    pylon_graph = graph.graph()
    max_width = WIDTH
    arm_width = WIDTH * 1.5
    mid_width = 2000
    height = HEIGHT
    centre_ids = []
    all_brace_ids = []

    def centre(t):
        pt_a, pt_b = section['base'], section['top']
        retval = (lambda t: interpolate(t, (pt_a, pt_b)))(t)
        return retval

    def brace(t):
        pt_a, pt_b = section['base_offset'], section['top_offset']
        retval = (lambda t: pt_plus_pt((lambda t: interpolate(t, (pt_a, pt_b)))(t),
                                       (lambda t: centre(t))(t)))(t)
        return retval

    def arm(t):
        retval = (lambda t: pt_plus_pt((lambda t: section['arm_offset'])(t), 
                                   (lambda t: pt_plus_pt(centre(t),[0,0,0]))(t)))(t)
        return retval

    def cross_brace(base_id, t):
        n = 2
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

    def create_arm(all_brace_ids, t):
        if len(all_brace_ids) > 1:
            arm_ids = []
            startpoint = arm(t)
            pylon_graph.add_unique_node(startpoint, 'line')
            for brace_id in all_brace_ids:
                endpoint = pylon_graph.node[brace_id]['xyz']
                arm_points = subdivide_line(startpoint, endpoint, 2)
                arm_ids.extend(pylon_graph.add_nodes(arm_points, 'arm', True)) 
            pylon_graph.connect_neighbours(arm_ids, 5000, True)


#########

            print "creating arm"
            xyz=arm(t)
            arm_id=pylon_graph.add_unique_node(xyz,'arm')
            for brace_id in all_brace_ids[0:2]:
                print "id",brace_id
                pylon_graph.add_edge(arm_id, brace_id)

    def make_section(add_lines=False):
        brace_ids = []
        for i in range(section['subsecs'] + 1):
            t = i / float(section['subsecs'])
            node_id = pylon_graph.add_unique_node(centre(t), 'centre')
            centre_ids.append(node_id)
            brace_ids.extend(cross_brace(node_id, t))
                
        if add_lines:
            lines = 3
            brace_set = list(set(brace_ids))
            print "brace_set", brace_set
            #connectors = random.sample(brace_set, 6)
            connectors = brace_set[0:6]
            print "connectors", connectors
            connectors.sort()
            connect_iter = iter(connectors)
            for i in range(lines):
                t = 0.3+i / float(lines)
                link_a = connect_iter.next()
                link_b = connect_iter.next()
                create_arm([link_a, link_b], t)

    section = {'base': [0, max_width, 0],'top': [0, mid_width, height * 0.4],
               'base_offset':  [max_width,0,0], 'top_offset': [mid_width,0,0],
               'arm_offset':[arm_width, -mid_width, 0.0], 'subsecs': 5}    
    make_section()

    section = {'base': [0 , mid_width, height * 0.4],'top': [0, mid_width, height * 0.8],
               'base_offset':  [mid_width,0,0], 'top_offset': [mid_width,0,0],
               'arm_offset':[arm_width, -mid_width, 0.0], 'subsecs': 5}
    make_section(True)

    section = {'base': [0 , mid_width, height * 0.8],'top':  [0, 0, height],
               'base_offset':  [mid_width,0,0], 'top_offset': [0,0,0],
               'arm_offset':[arm_width, -mid_width, 0.0], 'subsecs': 3}
    make_section()

    #pylon_graph.connect_nodes(centre_ids)
    pylon_graph.connect_nodes(all_brace_ids)
    rotated_graph = pylon_graph.copy_and_rotate_around_xy_plane(pylon_graph,180)
    mirror_graph = pylon_graph.copy_and_offset_with_mirror(rotated_graph, [0, 0, 0], True)
    full_graph = pylon_graph.copy_and_rotate_around_xy_plane(mirror_graph, 90)
    final_graph = pylon_graph.sanitise_pylon(full_graph, mid_width)
    pylon_graph.replace_graph(final_graph)
    return pylon_graph

def constrained_offset_graph(length=10000, levels=10):
    """build pylon using offset function. Add line labels for
    constraint checking"""
    LEVELS = levels
    LENGTH = length
    pylon_graph = graph.pylon_graph()
    base = square(ORIGIN, LENGTH)
    base_ids = pylon_graph.add_nodes(base, "line")
    pylon_graph.connect_neighbours(base_ids, LENGTH)
    all_ids = []
    for i in range(LEVELS):
        level = offset(base, LENGTH * i, "z")
        if i == 10:
            level_ids = pylon_graph.add_nodes(level, "line")
        else:
            level_ids = pylon_graph.add_nodes(level, "level" + str(i))
        all_ids.extend(level_ids)
    pylon_graph.connect_neighbours(all_ids, LENGTH)
    return pylon_graph

def constraint_test():
    """ Verify constraint checking methods."""
    import itertools, sys

    show_analysis = False
    #Generated via grammar
    gr = grammar.Grammar('grammars/test_constraints.bnf')
    inputs = ([1 for _ in range(100)], [ i%3 for i in range(100)])
    for _input in inputs: 
        output = gr.generate(_input)
        azr = analyser.Analyser('test',output['phenotype'],True)
        try:
            azr.create_graph()
        except ValueError as e:
            print(__name__, "ERROR", _input, e)
            continue
        azr.parse_graph()
        azr.apply_stresses()
        azr.create_slf_file()
        azr.test_slf_file()
        azr.parse_results()
        azr.print_stresses()
        if show_analysis:
            azr.show_analysis()
            
    #Fixed generated
    lengths = (1000, 10000)
    levels = (5, 10)
    for length_idx, level_idx in itertools.permutations([0,1]):
        try:
            GRAPH = constrained_offset_graph(lengths[length_idx],
                                             levels[length_idx])
        except ValueError as e:
            print(__name__, "ERROR", lengths[length_idx], levels[length_idx], e)
            continue
        GRAPH.save_graph("pylon")
        print "nodes:", GRAPH.number_of_nodes()
        print "edges", GRAPH.number_of_edges()
    #will it blend?
        azr = analyser.Analyser('test',"moo",True)
        azr.my_graph = GRAPH
        azr.parse_graph()
        azr.apply_stresses()
        azr.create_slf_file()
        azr.test_slf_file()
        azr.parse_results()
        azr.print_stresses()
        if show_analysis:
            azr.show_analysis()

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
    # azr.parse_graph()
    # azr.apply_stresses()
    # azr.create_slf_file()
    # azr.test_slf_file()
    # azr.parse_results()
    # azr.print_stresses()
    # azr.show_analysis()

if __name__ == '__main__':
    main()
