import binascii

with open("IMG_1416.mat", "rb") as f:
    byte = f.read(1)
    while byte:
        # Do stuff with byte.
        byte = f.read(1)
        x = binascii.unhexlify(byte)
        print x
