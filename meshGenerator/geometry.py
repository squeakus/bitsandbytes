#!/usr/bin/env python

# geometry.py
# jamesmichaelmcdermott@gmail.com
# GPL

# A little module for calculating geometrical stuff, like distance
# from a point to a line, whether a point is inside a polygon, and so
# on.

from math import *
import random
import copy

# NEW METHODS FOR BRIDGE GRAMMAR
# reflects points on a list through their respective X axes and returns a list
def mirror(pts):
    retval = list()
    for pt in pts:
        inverse1,inverse2,inverse3 = pt[0] * (-1),pt[1], pt[2]
        inverse = inverse1,inverse2,inverse3
        retval.append(inverse)
    pts.reverse()
    for pt in pts:
        retval.append(pt)
    return retval

def interpolate(p, xy):
        p = 1 - p
        x, y = xy
        x0, y0, z0 = x
        x1, y1, z1 = y
        return [x0 * p + x1 * (1 - p), y0 * p + y1 * (1 - p), z0 * p + z1 * (1 - p)]

# get the dot-sum of pt0 and pt1
def pt_plus_pt(pt0, pt1):
    return dot_operation(pt0, pt1, lambda x, y: x + y)

#generates a sinusoid between two points, returns a list
def sinusoid(ptA, ptB, height, length):
    retval =list()
    for i in range(ptA[0],ptB[0]):
        point = i, ptA[1], sin(i/float(length)) * height
        retval.append(point)
    return retval
    
#generates a sinusoid between two points, returns a list
def cosine(ptA, ptB , height, length):
    retval =list()
    for i in range(ptA[0],ptB[0]):
        point = i, ptA[1], cos(i/float(length)) * height
        retval.append(point)
    return retval

# shifts all z values up until theyare all positive, returns a list
def shiftValue(pts):
    minVal = 0;
    retval = list()
    for pt in pts:
        if pt[2] < minVal:
            minVal = pt[2]
    if minVal < 0:
        for pt in pts:
            shiftPoint = pt[0] , pt[1], pt[2] - minVal
            retval.append(shiftPoint)
        return retval    
    else:
        return pts

#flips a curve upside down for the bridge
def invert(pts,height):
    retval = list()
    for pt in pts:
      inverted = pt[0],pt[1],height - pt[2]
      retval.append(inverted)
    return retval


#takes a list and reflects any negative vals through the z axis
def absValue(pts):
    retval = list()
    for pt in pts:
        if pt[2] < 0:
            absPoint = pt[0] , pt[1], 0 - pt[2]
            retval.append(absPoint)
        else:            
            retval.append(pt)
    return retval    

# This method takes a list of points (tuples) and an offset tuple
# and returns a new list of points (tuples) offset by the values
# given in the offset tuple.
def offset_list(pts, offset):
    newList = []
    for pt in pts:
        newTup = (pt[0] + offset[0], pt[1] + offset[1], pt[2] + offset[2])
        newList.append(newTup)
    return newList

def euclidean_distance(p, q):
    return sqrt(sum([(p[i] - q[i]) **2 for i in range(len(p))]))


# fn is the "carrier function". fn(t) gives a point "in the centre of"
# the spiral. spiral() returns a single point on the outside of the
# spiral depending on the time parameter t. That is, call this
# multiple times with different values of t (same values for
# everything else) and you'll generate a spiral around the carrier
# curve.
def spiral(t, radius, initial_phase, revs, fn):
    epsilon = 0.001
    if t <= 1.0 - epsilon:
        pt0 = fn(t)
        pt1 = fn(t + epsilon)
    else:
        pt0 = fn(t - epsilon)
        pt1 = fn(t)
    phase = initial_phase + 2 * pi * t * revs
    x = disk_at_pt0_perp_to_line_pt0pt1(phase, pt0, pt1, radius)
    return x
    

# calculating a disk in the plane perpendicular to the line between two points:
# from http://local.wasp.uwa.edu.au/~pbourke/geometry/disk/
def disk_at_pt0_perp_to_line_pt0pt1(theta, pt0, pt1, r):

    rnorm, snorm = get_orthonormal_vectors(pt0, pt1)

    Qx = pt0[0] + r * cos(theta) * rnorm[0] + r * sin(theta) * snorm[0]
    Qy = pt0[1] + r * cos(theta) * rnorm[1] + r * sin(theta) * snorm[1]
    Qz = pt0[2] + r * cos(theta) * rnorm[2] + r * sin(theta) * snorm[2]

    return (Qx, Qy, Qz)

# get the four corners of a square *centered* at pt0
# such that the square lies in the plane perpendicular
# to the line from pt0 to pt1
def square_at_pt0_perp_to_line_pt0pt1(pt0, pt1, side):
    return rect_at_pt0_perp_to_line_pt0pt1(pt0, pt1, side, side)

# get the four corners of a rectangle *centered* at pt0
# such that the rectangle lies in the plane perpendicular
# to the line from pt0 to pt1. Rectangle has sides of given size.
def rect_at_pt0_perp_to_line_pt0pt1(pt0, pt1, side1, side2):
    half1 = side1 / 2.0
    half2 = side2 / 2.0
    rnorm, snorm = get_orthonormal_vectors(pt0, pt1)

    # we return these points in a strange order, perhaps,
    # to satisfy makeBoard() in render.py
    return [
        pt_minus_pt(pt_minus_pt(pt0, scale(rnorm, half1)), 
                    scale(snorm, half2)),
        pt_minus_pt(pt_plus_pt(pt0, scale(rnorm, half1)), 
                    scale(snorm, half2)),
        pt_plus_pt(pt_minus_pt(pt0, scale(rnorm, half1)), 
                   scale(snorm, half2)),
        pt_plus_pt(pt_plus_pt(pt0, scale(rnorm, half1)), 
                   scale(snorm, half2)),
        ]


# given two points, find two unit vectors which are orthogonal to the
# line between them. we use a couple of hacks to make sure they're
# nicely aligned to axes.
def get_orthonormal_vectors(pt0, pt1):

    if sum(map(abs, pt_minus_pt(pt0, pt1))) < 0.000001:
        # Two points are coincident: any two orthogonal unit vectors 
        # will do.
        return ((1, 0, 0), (0, 1, 0))

    # which way around should the results be? Want the vector which
    # will be used for short side of beam's cross-section to have the
    # zero z-component. Swap them around if necessary.
    swap_results = False
    if pt0[2] == pt1[2]:
        # in this condition, our heuristic below won't work. use a
        # special case. Arbitrary distant point.
        P = [pt1[0] + 100, pt1[1] + 57, pt0[2]]
        swap_results = False
    elif pt0[0] == pt1[0] and pt0[1] == pt1[1]:
        # same x and y values, differ only in z. another special case.
        return ((1, 0, 0), (0, 1, 0))
    else:
        # Set P to be pt1, dropped perpendicularly to the plane of pt0.
        # When we take R as being orthogonal to (pt1-pt0) and to (P-pt0),
        # the result is that R has a zero z-component. This means that the
        # beam's cross-section won't be canted.
        swap_results = True
        P = [pt1[0], pt1[1], pt0[2]]

    while True:
        # Then calculate R and S as cross-products => they're orthogonal to the line
        R = cross_product(pt_minus_pt(P, pt0), pt_minus_pt(pt1, pt0))
        S = cross_product(R, pt_minus_pt(pt1, pt0))

        try:
            # Then normalise them. If R is zero this gives a
            # ZeroDivisionError. That happens if we're unlucky when
            # choosing P. So alter P. (Then we won't have the nice
            # axis-alignment)
            rnorm = normalised(R)
            snorm = normalised(S)
            break
        except ZeroDivisionError:
            print "Caught a zerodivisionerror. pt0, pt1, P, R, S:", pt0, pt1, P, R, S
            for i in range(3):
                P[i] += random.random()
            print "Now trying pt0, pt1, P, R, S:", pt0, pt1, P, R, S
            print "swap_results == " + str(swap_results)
    if swap_results:
        return snorm, rnorm
    else:
        return rnorm, snorm


# get the vector from pt0 to pt1
def pt_minus_pt(pt0, pt1):
    return dot_operation(pt0, pt1, lambda x, y: x - y)

# use this to do (eg) (1, 1, 1) + (4, 4, 6)
# pass in lambda x, y: x + y as the operation.
def dot_operation(pt0, pt1, fn):
    return [fn(x0, x1) for x0, x1 in zip(pt0, pt1)]

# scale a vector by a scaling factor
def scale(pt, factor):
    return [x * factor for x in pt]

def cross_product(pt0, pt1):
    i = pt0[1] * pt1[2] - pt0[2] * pt1[1]
    j = pt0[2] * pt1[0] - pt0[0] * pt1[2]
    k = pt0[0] * pt1[1] - pt0[1] * pt1[0]
    return [i, j, k]

def vector_length(pt):
    return sqrt(sum([x ** 2 for x in pt]))

def normalised(pt):
    s = vector_length(pt)
    return [pt[0] / s, pt[1] / s, pt[2] / s]
    

# Check whether a point is inside a polygon. I think the vertices have
# to be given in the anti-clockwise order, so we always want pt to be
# "on the left" of the line between the current vertex and the next
# vertex. This only works for convex polygons. They don't have to be
# regular though.
def inside_polygon(pt, vertices):
    n = len(vertices)
    for i in range(n):
        if is_on_the_right(pt, (vertices[i], vertices[(i + 1) % n])):
            return False
    return True

# "On the right" means as we go from vertex 0 to vertex 1.
def is_on_the_right(pt, vertices):
    # if the line is vertical, then pt is "on the right" in the v0->v1
    # direction if 2nd vertex has greater y-value than the 1st and pt
    # is actually "on the right" (ignoring direction). pt is "on the
    # right" in the v0->v1 sense if it's actually on the left and v0
    # is above v1. Argh it's hard!
    if vertices[0][0] == vertices[1][0]:
        if vertices[1][1] > vertices[0][1]:
            return pt[0] > vertices[0][0]
        else:
            return pt[0] < vertices[0][0]
    # if the 2nd vertex has a smaller x-value than the 1st, then pt is
    # "on the right" *iff* it's *above* the line.
    above = above_line(pt, vertices)
    if (vertices[1][0] < vertices[0][0]):
        return above
    else:
        return not above

# Is a point (2-tuple) above a line (2-tuple of 2-tuples)?
def above_line(pt, vertices):
    # if the line is vertical, the function is not defined
    if vertices[0][0] == vertices[1][0]:
        raise ValueError, "above_line() not defined for vertical lines"
    # if the line is horizontal, c won't be defined
    if vertices[0][1] == vertices[1][1]:
        return (pt[1] >= vertices[0][1])
    # otherwise use line equation: y - y1 = m(x - x1)
    # m is slope, c is x-intercept
    # m = (y2 - y1) / (x2 - x1)
    m = (vertices[1][1] - vertices[0][1]) / \
        float(vertices[1][0] - vertices[0][0])
    # deriving from the line equation: when y = 0 (x-intercept),
    # x = x1 - y1 / m
    c = vertices[0][0] - vertices[0][1] / m
    y_val_at_x = m * (pt[0] - vertices[0][0]) + vertices[0][1]
    return (pt[1] >= y_val_at_x)


# Distance from a point to a line. pt is a 2-tuple, (x0, y0). Line is
# a 2-tuple of two pts ((x1, y1), (x2, y2)).
def point_line_dist(pt, line):
    x0, y0 = pt
    x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
    if (x2 == x1):
        return fabs(x1 - x0)
    
    A = (y2 - y1) / (x2 - x1)
    B = -1
    C = y1 - x1 * (y2 - y1) / (x2 - x1)
    top = fabs(x0 * A + y0 * B + C)
    bottom = sqrt(A * A + B * B)
    return top / bottom

# pt, line as above.
def point_line_intersection(pt, line):
    p1, p2 = line
    if p1 == p2:
        raise ValueError
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = pt

    d = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / (d * d)

    return (x1 + u * (x2 - x1), y1 + u * (y2 - y1))


# Given an angle in radians, says which of the six inscribed triangles
# in a hexagon (centred on the origin) the angle brings us to (going
# anticlockwise, counting from the rightmost point).
def triangle_no(angle):
    return (int) (angle * 6 / (2 * pi))


# pt is a 2-tuple. Returns an angle in radians.
def angle(pt):
    x, y = pt
    try:
        if x >= 0.0 and y >= 0.0:
            return asin(y / sqrt(x * x + y * y))
        elif x < 0.0 and y >= 0.0:
            return pi - asin(y / sqrt(x * x + y * y))
        elif x < 0.0 and y < 0.0:
            return pi + asin(-y / sqrt(x * x + y * y))
        else:
            return 2 * pi - asin(-y / sqrt(x * x + y * y))
    except ZeroDivisionError:
        return 0.0
        
# Given the vertices of a hexagon plus the centre point, plus a point
# in the hexagon, calculate the proportions (weights) corresponding to
# the vertices.
def get_hex_proportions(hex_pts, pt):
    # which triangle is the point in?
    ang = angle(pt)
    tri = triangle_no(ang)

    # distances from pt to the three lines comprising that triangle
    dists = [0.0] * 7
    dists[tri] = point_line_dist(pt, (hex_pts[(tri+1)%6], hex_pts[6]))
    dists[(tri+1)%6] = point_line_dist(pt, (hex_pts[tri], hex_pts[6]))
    # centre proportion is dists[6]
    dists[6] = point_line_dist(pt, (hex_pts[tri], hex_pts[(tri+1)%6]))
    s = sum(dists)
    return [d / s for d in dists]


# Given the vertices of a triangle, and a point within that triangle,
# calculate the proportions (weights) corresponding to the vertices.
def get_tri_proportions(tri_pts, pt):
    # distance from pt to the three lines
    dists = [point_line_dist(pt, (tri_pts[i], tri_pts[(i+1)%3]))
             for i in range(3)]
    # scale the distances so that they add to 1.0
    s = sum(dists)
    return [d / s for d in dists]


# Pass in the side of the equilateral triangle and this returns its
# height.
def triangle_height(x):
    return sqrt(pow(x, 2) - pow(x / 2.0, 2))

# Given a point in n-space, determine whether it's inside the unit
# n-cube.
def inside_unit_n_cube(pt):
    for pti in pt:
        if pti < 0.0 or pti > 1.0:
            return False
    return True

# The L1 distance just takes the minimum of the distances in each
# dimension.
def L1_distance_to_n_cube_boundary(p):
    return min([min([abs(pti), abs(1.0 - pti)]) for pti in pt])

# Given two 01-list presets (centre prs and a distance prs), find the
# point where the projection from one to the other meets the boundary.
# There are better ways to do this, but it's not worth the effort here.
def projection_to_boundary(cp, dp):
    if cp == dp:
        raise ValueError, "Input points were identical, can't project"

    v = [dpi - cpi for cpi, dpi in zip(cp, dp)]
        
    # The point returned will be short of the boundary by up to this much:    
    desired_accuracy = 0.001 
    old_pt = dp[:]
    c_accuracy = 1.0
    scale = 0.5
    
    while c_accuracy > desired_accuracy and scale > 0.00001:
        print "in while, c_accuracy", c_accuracy, "scale", scale
        # Make a new point by adding a little bit (scale) to the last
        # pt inside the cube.

        new_pt = [opi + vi * scale for opi, vi in zip(old_pt, v)]

        # If it's still inside, continue from that point; else
        # decrease the amount to be added.
        if inside_unit_n_cube(new_pt):
            old_pt = new_pt
            c_accuracy = L1_distance_to_n_cube_boundary(old_pt)
        else:
            scale /= 2.0

    return old_pt
    
        


# Make a new 2-tuple point inside the given polygon. The problem is
# to avoid generating a point outside the polygon. There are clever
# ways of doing it, using the right distribution, but simply looping
# is an ok solution.
def generate_random_pt_inside_polygon(poly):
    maxx = max([pt[0] for pt in poly])
    minx = min([pt[0] for pt in poly])
    maxy = max([pt[1] for pt in poly])
    miny = min([pt[1] for pt in poly])

    outside = True
    while outside:
        pt = (random.uniform(minx, maxx), random.uniform(miny, maxy))
        if inside_polygon(pt, poly):
            outside = False
    return pt

# The vertices of a "unit" hexagon, starting with the right-most point
# and going anti-clockwise (in the direction of positive theta in the
# unit circle (cos(theta), sin(theta))).
def hex_vertices(size=1.0):
    
    return [
        (size, 0.0),
        (0.5 * size, triangle_height(size)),
        (-0.5 * size, triangle_height(size)),
        (-size, 0.0),
        (-0.5 * size, -triangle_height(size)),
        (0.5 * size, -triangle_height(size)),
        ]

# the vertices of a unit equilateral triangle, anti-clockwise. Starts
# at origin, goes clockwise.
def tri_vertices(size=1.0):
    return [(0, 0), 
            (1.0 * size, 0), 
            (0.5 * size, triangle_height(1.0 * size))]


def dot_product(a, b):
    return sum([ai * bi for ai, bi in zip(a, b)])

def mag(a):
    return sqrt(sum([ai * ai for ai in a]))

def angle_between_two_vectors(a, b):
    try:
        # This can also raise a ZeroDivisionError: caller must catch
        # it and decide what to do with it.
        return acos(dot_product(a, b) / (mag(a) * mag(b)))
    except ValueError:
        # Floating-point rounding errors can make the X in
        # acos(X) slightly over 1.0 or slightly under -1.0: this
        # will raise a ValueError. When we catch this, check whether
        # the argument is positive or negative, and return the exact
        # value.
        if dot_product(a, b) > 0.0:
            return acos(1.0)
        else:
            return acos(-1.0)

def angle_between_two_vectors_given_base_pt(base, a, b):
    av = [ai - basei for ai, basei in zip(a, base)]
    bv = [bi - basei for bi, basei in zip(b, base)]
    return angle_between_two_vectors(av, bv)


