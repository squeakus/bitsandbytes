"""Writing a sift header and descriptor and then reading it again."""
import struct

outfile = open('out.bin', 'wb')
binary = struct.pack('cccccccc', 'S', 'I', 'F', 'T', 'V', '4', '.', '0')
outfile.write(binary)
binary = struct.pack('iii', 5, 2, 3)
outfile.write(binary)
binary = struct.pack('BBBB', 5, 2, 3, 1)
outfile.write(binary)
outfile.close()


with open('out.bin', 'rb') as f:
    content = f.read()
    name = str(struct.unpack('cccc', content[:4]))
    version = str(struct.unpack('cccc', content[4:8]))
    npoints = struct.unpack('i', content[8:12])[0]
    print "name:", name, "version", version, "npoints", npoints
    descriptor = struct.unpack('BBBB', content[20:24])
    print descriptor
