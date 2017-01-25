"""This program converts GOCAD Tsurf files to PLY files."""
import sys


def main(filename):
    """Given a filename pull out the verts and tris."""
    print "now converting", filename, "to out.ply"
    infile = open(filename, 'r')
    vertices = []
    triangles = []
    for line in infile:
        if line.startswith('VRTX'):
            line = line.split()
            x = line[2]
            y = line[3]
            z = str(-float(line[4]))
            vertices.append([x, y, z])

        elif line.startswith('TRGL'):
            line = line.split()
            a = line[1]
            b = line[2]
            c = line[3]
            triangles.append([a, b, c])

    print "verts:", len(vertices)
    print "tris:", len(triangles)
    write_ply(vertices, triangles)


def write_ply(verts, tris):
    """Output the verts and tris in ply format."""
    ply_file = open('out.ply', 'w')

    vertcnt = len(verts)
    tricnt = len(tris)

    ply_file.write('ply\n')
    ply_file.write('format ascii 1.0\n')
    ply_file.write('comment : created by Jonathan Byrne UCD\n')
    ply_file.write('element vertex %d\n' % vertcnt)
    ply_file.write('property float x\n')
    ply_file.write('property float y\n')
    ply_file.write('property float z\n')
    ply_file.write('element face %d\n' % tricnt)
    ply_file.write('property list uchar int vertex_indices\n')
    ply_file.write('end_header\n')

    for vert in verts:
            x, y, z = vert
            ply_file.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')

    for tri in tris:
        a, b, c = tri
        ply_file.write('3 ' + a + ' ' + b + ' ' + c + '\n')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "usage: python %s <filename>" % sys.argv[0]
        exit()
    main(sys.argv[1])
