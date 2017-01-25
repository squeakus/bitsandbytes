with open("test.bin", "rb") as f:
    byte = f.read(1)
    while len(byte) > 0:
        print '{0:08b}'.format(ord(byte))
        byte = f.read(1)
