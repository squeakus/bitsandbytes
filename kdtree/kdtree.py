from math import sqrt
def main():
    target = (2,0)
    points = [(2,1), (4,2), (1,3), (9,4), (8,5), (2,6), (6,7)]

    my_tree = kd_tree(points, 0)
    nearest = search_tree(my_tree, target, 0)
    print "nearest neighbour:", nearest
    print "distance:", distance(target, nearest)


def kd_tree(points, depth):
    if len(points) == 1:
        return points[0]

    else:
        # sort using either x or y
        axis = depth % 2
        points.sort(key=lambda tup: tup[axis])

        mid = len(points)//2
        median = points[mid][axis]
        before = points[:mid]
        after = points[mid:]

        tree_left = kd_tree(before, depth+1)
        tree_right = kd_tree(after, depth+1)

        return median, tree_left, tree_right

def search_tree(tree, point, depth):
    if len(tree) == 2:
        return tree
    else:
        axis = depth % 2
        med = tree[0]
        val = point[axis]
        if val < med:
            tree = search_tree(tree[1], point, depth+1)
        else:
            tree = search_tree(tree[2], point, dpeth+1)
        return tree

def distance(p, q):
    return sqrt(sum([(p[i] - q[i]) **2 for i in range(len(p))]))


if __name__=='__main__':
    main()
