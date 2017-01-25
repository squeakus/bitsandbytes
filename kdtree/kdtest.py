def main():
    #points = [9,9,9,2,3,5,7,9,9,9,9,9,5,4,3,3,2]
    
    #points = [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7)]
    points = [(2,1), (4,2), (1,3), (9,4), (8,5), (2,6), (6,7)]
    #points = [(1,6), (1,7), (1,3), (1,4), (1,5), (1,1), (1,2)]
    print "median:", get_median(points)


def get_median(points):
    points.sort(key=lambda tup: tup[0])
    print "sorted", points
    mid = len(points)//2
    median = points[mid]
    before = points[:mid]
    after = points[mid:]
    
    
    print points
    print before
    print after
    return median

if __name__ == '__main__':
    main()
