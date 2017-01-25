import struct

binary = [1234, 5678, -9012, -3456]
with open('out.bin', 'wb') as f:
    for b in binary:
        f.write(struct.pack('i', b)) #or whatever format you need


with open('out.bin', 'rb') as f:
    content = f.read()
    for b in content:
        print(b)
    print(struct.unpack('iiii', content))


