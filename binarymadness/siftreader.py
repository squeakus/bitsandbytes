"""reads changchangs format sift files."""
import struct

features = []
descriptors = []

with open('IMG_1416.sift', 'rb') as f:
    content = f.read()
    # read the header
    offset = 20
    # name, version, npoint, notsure,dimension =
    print(struct.unpack('cccc cccc iii', content[:offset]))  # int is 4 bytes
    name, version, npoints, notsure, dim = struct.unpack('iiiii',
                                                         content[:offset])
    print "name, version, npoint, notsure,dimension"

    # DONT ASK ME WHY BUT THEY REVERSE THE X AND Y
    # descriptor location and orientation
    for point in range(npoints):
        # print "point:", point, "offset:", offset
        y, x, color, scale, orient = struct.unpack('fffff',
                                                   content[offset:offset+20])
        offset += 20
        features.append((y, x, color, scale, orient))

        if point < 4:
            print x, y, color, scale, orient

    # descriptor info
    for point in range(npoints):
        # print "point:", point, "offset:", offset
        descriptor = struct.unpack('B'*128, content[offset:offset+128])
        descriptors.append(descriptor)
        offset += 128

        if point < 4:
            print descriptor

print "features:", len(features), "descriptors", len(descriptors)

with open('out.sift', 'wb') as outfile:
    binary = struct.pack('cccccccc', 'S', 'I', 'F', 'T', 'V', '4', '.', '0')
    outfile.write(binary)
    binary = struct.pack('iii', npoints, notsure, dim)
    outfile.write(binary)

    for feat in features:
        binary = struct.pack('fffff', *feat)
        outfile.write(binary)

    for desc in descriptors:
        binary = struct.pack('B'*128, *desc)
        outfile.write(binary)
