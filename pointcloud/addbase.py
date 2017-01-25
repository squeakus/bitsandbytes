import numpy as np
def main():
    points = parse_ply('subsample.ply')
    print get_stats(points)
    basepoints = build_base(points)
    create_ply(basepoints, "base.ply")

def create_ply(points, filename):
    print 'Generating ply file', filename
    print "no of points", len(points)
    
    ply_file = open(filename, 'wt')
    ply_file.write('ply\n')
    ply_file.write('format ascii 1.0\n')
    ply_file.write('comment : created by depth2cloud\n')
    ply_file.write('element vertex %d\n' % len(points))
    ply_file.write('property float x\n')
    ply_file.write('property float y\n')
    ply_file.write('property float z\n')
    ply_file.write('end_header\n')

    for point in points:
        x, y, z = point
        ply_file.write("%f  %f  %f\n" % (x, y, z))
    ply_file.close()
    
def build_base(points):
    stats = get_stats(points)
    xcoords = np.arange(stats['minx'],stats['maxx'],0.05)
    ycoords = np.arange(stats['miny'],stats['maxy'],0.05)

    for x in xcoords:
        for y in ycoords:
            points.append((x,y,stats['minz']))

    return points
    
def parse_ply(filename):
    plyfile = open(filename, 'r')
    points = []
    vertcount = 0
    line = ""

    while not line.startswith("end_header"):
        line = plyfile.readline().rstrip()
        if line.startswith("element vertex"):
            line = line.lstrip("element vertex")
            vertcount = int(line)

    for _ in range(vertcount):
        line = plyfile.readline()
        line = line.split(' ')
        x,y,z = float(line[0]),float(line[1]),float(line[2])
        points.append((x,y,z))
    plyfile.close()
    return points


def get_stats(points):
    inf, ninf = float('inf'), float('-inf')
    stats = {'maxx':ninf, 'minx':inf,
              'maxy':ninf, 'miny':inf,
              'maxz':ninf, 'minz':inf}

    for point in points:
        x, y, z = point
        if x > stats['maxx']: stats['maxx'] = x
        if x < stats['minx']: stats['minx'] = x
        if y > stats['maxy']: stats['maxy'] = y
        if y < stats['miny']: stats['miny'] = y
        if z > stats['maxz']: stats['maxz'] = z
        if z < stats['minz']: stats['minz'] = z
    stats['xrange'] = stats['maxx'] - stats['minx']
    stats['yrange'] = stats['maxy'] - stats['miny']
    stats['zrange'] = stats['maxz'] - stats['minz']
    return stats

if __name__=='__main__':
    main()

