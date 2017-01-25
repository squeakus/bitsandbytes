"""Constraints for graph
Copyright (c) 2010 Jonathan Byrne, Erik Hemberg and James McDermott
Hereby licensed under the GNU GPL v3."""

#TODO More effecient way than comparing all nodes
#TODO Decide which data structure the functions work with. The
#networkx graph node or the node created from analyser.py
#TODO insulator limit for suspension tower is 6250
from math import sin, sqrt, radians, cos
from random import randrange, randint

LINE_POINT_LIMIT = 8000
INSULATOR_LIMIT = 5200 
STRUCT_LIMITS = (3100, 1800)
LINE_LIMIT = INSULATOR_LIMIT + STRUCT_LIMITS[0]
STRUCT_ANGLE = radians(35)
STRUCT_ANGLE_RNGS = ((INSULATOR_LIMIT + STRUCT_LIMITS[0]) * \
                              sin(STRUCT_ANGLE), 
                          INSULATOR_LIMIT * sin(STRUCT_ANGLE) + \
                              STRUCT_LIMITS[1])
CONDUCTOR_SAG = 12050
GROUND_CLEARANCE = 8100
MIN_HEIGHT = CONDUCTOR_SAG + GROUND_CLEARANCE + INSULATOR_LIMIT

def check_structure_constraint(my_nodes, line_nodes):
    """Verify that the structure points are not violating the structure
    constraints.
    
    The structure constraints are that each structure point must be at
    least 3.1m away from the line point."""
#    print(__name__,"check_structure_constraint")
    #Compare structure points
    for node in my_nodes:
        xyz_0 = (node['x'], node['y'], node['z'])
        for line_node in line_nodes:
            if line_node != node:
                xyz_1 = (line_node['x'], line_node['y'], line_node['z'])
                insulator_pt = (xyz_0[0], xyz_0[1] - INSULATOR_LIMIT, 
                                   xyz_0[2])
                dist = euclidean_distance(xyz_0, xyz_1)
#                print(__name__, "check_structure_constraint",
#                      dist, node, line_node)
                #y-axis
                if xyz_1[1] > xyz_0[1]:
                    #3.1m limit, above the line pt 
                    if dist < STRUCT_LIMITS[0]:
                        return False
                else:
                    #Check the x-axis 3.1 radi
                    #TODO take into account the increased y from the swing
                    if ((insulator_pt[0] - STRUCT_ANGLE_RNGS[0]) < xyz_1[0] or \
                            (insulator_pt[0] + STRUCT_ANGLE_RNGS[0]) > xyz_1[0]):
                        if dist < STRUCT_LIMITS[0]:
                            return False
                    else:
                        #Check the x-axis 1.8 radi                   
                        if ((xyz_0[0] - STRUCT_ANGLE_RNGS[1]) < xyz_1[0] or \
                                (xyz_0[0] + STRUCT_ANGLE_RNGS[1]) > xyz_1[0]):
                            if dist < STRUCT_LIMITS[1]:
                                return False
    return True
           
def check_insulator_constraint(my_nodes, my_graph):
    """Verify that the insulator is not violating any constraints.

    Insulator must be 5.2m"""
#    print(__name__,"check_insulator_constraint")
    #TODO This does not allow branching from the insulator
    for node in my_nodes:
        xyz = (node['x'], node['y'], node['z'])
        for adj_node in my_graph.neighbors_iter(int(node['id'])):
            if int(node['id']) != adj_node:
                xyz_adj = my_graph.get_node_data(adj_node)
                xyz_adj = (xyz_adj[0], xyz_adj[1], xyz_adj[2])
                dist = euclidean_distance(xyz, xyz_adj)
                print("CIC", node, xyz_adj, dist, (dist < INSULATOR_LIMIT))
                if dist < INSULATOR_LIMIT:
                    return False
    return True

def check_line_constraint(line_nodes):
    """Verify that the line points are not violating the line
    constraints.
    
    The line constraints are that each line point must be at least 8m
    away."""
#    print(__name__,"check_line_constraint")
    #Compare line points
    for i in range(len(line_nodes)):
        current_node = line_nodes[i]
        xyz = (current_node['x'], current_node['y'], current_node['z'])
        for j in range(i+1, len(line_nodes)):
            cmpr_node = line_nodes[j]
            dist = euclidean_distance(xyz, 
                                      (cmpr_node['x'], 
                                       cmpr_node['y'], 
                                       cmpr_node['z']))
#            print(__name__, "check_line_constraint", 
#                  dist, i, j, current_node, cmpr_node)
            if dist < LINE_LIMIT:
                return False
    return True

def euclidean_distance(origin, point):
    """Euclidean distance between origin and point"""
    diffs = [pow(x - y, 2) for x, y in zip(origin, point)]
    distance = sqrt(sum(diffs))
    return distance

def configure_line_points(width, heigth, npoints=6):
    """Line points are placed guaranteeing feasibility between points.

    Symetry specifies if the points are optimally mirrored and
    feasibility is guaranteed for the line limit. The coordinates (0,
    0, 0) start in the bottom left corner of the cube. Does not allow
    configuration along the y-axis. Width assumes the width of one
    side. The mirroring is done at the shortes allowed distance.

    TODO check incoming values and outgoing (contract)
    TODO allow assymetric placement"""
    placements = npoints/2
    coords = [[0,0,0] for _ in range(npoints)]
    #Optimal mirroring distance
    max_x = 0
    for i in range(placements):
        x = randrange(0,width)
        if x > max_x:
            max_x = x
        y = 0
        if i == 0:
            z = 0
        else:
            z = randrange(coords[i - 1][2] + LINE_LIMIT,
                          heigth-(LINE_LIMIT*(placements-(i+1))))
        coords[i] = (x, y, z)
    #Mirroring
    for i in range(placements, len(coords)):
        x = (2 * max_x) - coords[i - placements][0] + LINE_LIMIT
        y = coords[i - placements][1]
        z = coords[i - placements][2]
        coords[i] =  (x, y, z)
    print(__name__,"configure_line_points", coords)
    return coords

def optimal_line_placement(npoints=6):
    """Optimally place the lines.
    The optimla placement is each lin point being LINE_LIMIT away,
    0,0,0 is bottom left corner"""
    coords = [[] for _ in range(npoints)]
    for i in range(npoints):
        if i < npoints/2:
            coords[i] = (0, 0, i*LINE_LIMIT)
        else:
            j = i - npoints/2
            coords[i] = (coords[j][0] + LINE_LIMIT, coords[j][1], coords[j][2])
    coords = coords[0:3]
    return coords

def line_configuration(angle = -1, angle_1 = -1, extra_line_limit = 0, invert=False):
    """Line point configuration for an triangle rotated by
    angle.

    Angle -1 returns optimal_line_placement. Angle is the angle of the
    leg towards the mid z-point. The last point is a doubling of the
    mid z-point if angle_1 is 90. Returns 3 line points.
    """
    #TODO how well does the transformation to int work
    #TODO MIN_HEIGHT takes INSULATOR_LIMIT into account
    #TODO less hard coding
    min_coord = 0
    if angle == -1:
        new_coords = optimal_line_placement()[:3]
    else:
        line_limit = LINE_LIMIT + extra_line_limit
#        coords = equidistant_line_placement()[:3]
        new_coords = [(0, 0, 0)]
        angle_r = radians(angle)
        x = int(cos(angle_r) * line_limit)
        y = 0 
        z = int(sin(angle_r) * line_limit)
        new_coords.append((x, 
                           y, 
                           z)) 
        z_side = max(line_limit, z * 2)        
        #x-coord for point 1 and 3 are always 0
        if angle_1 == -1:
            new_coords.append((0, 
                               0, 
                               z_side)) 
        else:
            #Cannot be greater than 150
            angle_1 = min(angle_1, 150)
            angle_r = radians(angle_1)
            x = max(0, x + (int(cos(angle_r) * line_limit)))
            z = max(line_limit + z, z + (int(sin(angle_r) * line_limit)))
            new_coords.append((x, 
                               0, 
                               z)) 
        if invert:
            invert_x = (new_coords[0][0], new_coords[2][0])
            new_coords[0] = (invert_x[1], new_coords[0][1], new_coords[0][2])
            new_coords[2] = (invert_x[0], new_coords[2][1], new_coords[2][2])

    new_coords = [ (coord[0], coord[1], coord[2] + MIN_HEIGHT) for coord in new_coords]

    return new_coords

def equidistant_line_placement(extra_line_limit=0, stochastic=False, optimal_mirror=True):
    #TODO better handling of rounding of floats
    if optimal_mirror:
        mirror_limit = LINE_LIMIT
    else:
        mirror_limit = randint(LINE_LIMIT, (LINE_LIMIT + extra_line_limit))
    if stochastic:
        extra_line_limit = randint(1, extra_line_limit)
    line_limit = LINE_LIMIT + extra_line_limit
    #Increasing LINE_LIMIT to handle precision lost when rounding
    #float to int
    limit = line_limit + 2
    npoints = 6
    coords = [(0, 0, 0),
              (int((limit*sqrt(3))/2), 0, limit/2),
              (0, 0, limit)]
    coords = coords + \
        [(coord[0] + mirror_limit, coord[1], coord[2]) for coord in coords]
    #print(__name__, "equidistant_line_placement", coords)
    return coords

PROFILE = False

if __name__ == '__main__':
    import pylon
    #TODO proper unittest framework    
    if PROFILE:
        import cProfile, pstats
        profile_file = 'constraints_test'
        cProfile.run('pylon.constraint_test()')
        p = pstats.Stats(profile_file)
        p.strip_dirs().sort_stats(-1).print_stats()
        print('###### CUMULATIVE 10 #####')
        p.sort_stats('cumulative').print_stats(10)
        print('###### TIME 10 #####')
        p.sort_stats('time').print_stats(10)
    else:
#        pylon.constraint_test()
        configure_line_points(1000, 19001)
        configure_line_points(1000, 19001)
        equidistant_line_placement()
        equidistant_line_placement(1000)
        equidistant_line_placement(1000, True)
        equidistant_line_placement(1000, True, False)
        print("angle -1")
        line_configuration()
        angles = (10, 30, 0, 90)
        angles1 = (-1, 150, 90, 150)
        for angle in angles:
            print("--angle", angle)
            line_configuration(angle)
        for angle, angle1 in zip(angles, angles1):
            print("--angle", angle, angle1)
            line_configuration(angle, angle1)
        for angle, angle1 in zip(angles, angles1):
            print("--angle", angle, angle1, True)
            line_configuration(angle, angle1, 0, True)

