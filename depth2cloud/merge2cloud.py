import sys, pylab
from math import sin, cos, radians

def main():
    if len(sys.argv) != 3:
        usage()
    top = sys.argv[1]
    base = sys.argv[2]

    parser = CloudParser([top, base])
    parser.parse_image(top)
    parser.parse_image(base)
    parser.calc_ranges()
    print "RANGES", parser.ranges
    for idx, cloud in enumerate(parser.clouds):
        parser.clouds[idx] = parser.translate_points(cloud)

        parser.clouds[0] = parser.rotate_points(parser.clouds[0])
    parser.saveclouds()
    print 'Done!'

def usage():
    print '\ndepth2cloud <bin_file> <bin_file>\n'
    exit(0)


    
class CloudParser():
    def __init__(self, name_list):
        self.names = name_list
        self.scale_factor = 0.0021
        self.min_dist = -10
        self.clouds = []
        inf, ninf = float('inf'), float('-inf')
        self.ranges = {'maxx':ninf, 'minx':inf,
                       'maxy':ninf, 'miny':inf,
                       'maxz':ninf, 'minz':inf}

    def parse_image(self, imgfile):
        print 'Reading binary point cloud', imgfile
        raw = pylab.fromfile(imgfile, 'H')
        points = []

        # Compute depth (unit in cm) from raw 11-bit disparity value
        # According to ROS site
        depth = 100/(-0.00307*raw + 3.33)
        depth.resize(480, 640)

        # Convert from pixel ref (i, j, z) to 3D space (x,y,z)
        for i in range(480):
            for j in range(640):
                x = depth[i][j]
                z = (i - 480 / 2) * (x + self.min_dist) * self.scale_factor
                y = (640 / 2 - j) * (x + self.min_dist) * self.scale_factor

                #translate crazy depths
                if not x > 200:
                    if not x < 0:
                        points.append((x, y, z))
        self.clouds.append(points)

    def calc_ranges(self):
        for cloud in self.clouds:
            for point in cloud:
                x, y, z = point
                if x > self.ranges['maxx']: self.ranges['maxx'] = x
                if x < self.ranges['minx']: self.ranges['minx'] = x
                if y > self.ranges['maxy']: self.ranges['maxy'] = y
                if y < self.ranges['miny']: self.ranges['miny'] = y
                if z > self.ranges['maxz']: self.ranges['maxz'] = z
                if z < self.ranges['minz']: self.ranges['minz'] = z
        self.ranges['xrange'] = self.ranges['maxx'] - self.ranges['minx']
        self.ranges['yrange'] = self.ranges['maxy'] - self.ranges['miny']
        self.ranges['zrange'] = self.ranges['maxz'] - self.ranges['minz']

    def translate_points(self, oldpoints):
        newpoints = []
        for oldpoint in oldpoints:
            x, y, z = oldpoint
            y = y - self.ranges['miny'] # move to origin
            y = y - (self.ranges['yrange'] / 2) # make midpoint 0
            #invert z and move to origin
            z = self.ranges['zrange'] - (z - self.ranges['minz'])
            z = z - (self.ranges['zrange'] / 2) # make midpoint 0
            newpoints.append((x, y, z))
        return newpoints

    def rotate_points(self, oldpoints):
        deg = 16
        newpoints = []
        for point in oldpoints:
            x,y,z = point
            newx = (x * cos(radians(deg))) - (z * sin(radians(deg)))
            newz = (x * sin(radians(deg))) + (z * cos(radians(deg)))
            newpoints.append((newx, y, newz))
        return newpoints
            
    def saveclouds(self):
        print "namelist", self.names

        for idx, cloud in enumerate(self.clouds):
            self.create_ply(cloud, self.names[idx])

    def create_ply(self, points, filename):
        filename = filename.rstrip('.bin') + '.ply'
        print 'Generating ply file', filename
        print "no of points", len(points)

        ply_file = open(filename, 'wt')
        ply_file.write('ply\n')
        ply_file.write('format ascii 1.0\n')
        ply_file.write('comment : created from Kinect depth image\n')
        ply_file.write('element vertex %d\n' % len(points))
        ply_file.write('property float x\n')
        ply_file.write('property float y\n')
        ply_file.write('property float z\n')
        ply_file.write('end_header\n')
        for point in points:
            x, y, z = point
            ply_file.write("%f  %f  %f\n" % (x, y, z))
        ply_file.close()


if __name__ == '__main__':
    main()
