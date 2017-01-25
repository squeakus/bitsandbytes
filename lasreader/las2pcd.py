import liblas, vtk

def main():
    filename = "out.pcd"
    lascloud = liblas.file.File('316500_234500.las',mode='r')
    print "no of points:", len(lascloud)
    point = lascloud.read(0)
    color = point.get_color()
    print point.x, point.y, point.z, point.intensity
    print color.blue, color.red, color.green

    
def writepcd(filename, lascloud):
    
    
if __name__=='__main__':
    main()
