"""Jamming a bunch of values into a binary file."""

import struct
import binascii

values = (1, 'ab', 2.7)
# values = (1, 256)

s = struct.Struct('I 2s f')
packed_data = s.pack(*values)

print 'Original values:', values
print 'Format string  :', s.format
print 'Uses           :', s.size, 'bytes'
print 'Packed Value   :', binascii.hexlify(packed_data)

outfile = open('out.bin', 'wb')
outfile.write(packed_data)
outfile.close()


with open('out.bin', 'rb') as f:
    content = f.read()
    print "final result:", struct.unpack('I 2s f', content)
