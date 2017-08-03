"""Testing the svol format."""
import struct
import argparse
import numpy as np


def main():
    """Examines the contents of a sparse volume, (hard coded for depth 3)"""
    parser = argparse.ArgumentParser()
    parser.add_argument("svol",
                        help="the name of the vola file to open", type=str)
    args = parser.parse_args()

    with open(args.svol, "rb") as f:
        headersize = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('H', f.read(2))[0]
        mode = struct.unpack('B', f.read(1))[0]
        depth = struct.unpack('B', f.read(1))[0]
        crs = struct.unpack('I', f.read(4))[0]
        lat = struct.unpack('d', f.read(8))[0]
        lon = struct.unpack('d', f.read(8))[0]
        minx = struct.unpack('d', f.read(8))[0]
        miny = struct.unpack('d', f.read(8))[0]
        minz = struct.unpack('d', f.read(8))[0]
        maxx = struct.unpack('d', f.read(8))[0]
        maxy = struct.unpack('d', f.read(8))[0]
        maxz = struct.unpack('d', f.read(8))[0]

        # if mode mod 2 has a remainder it is using two bits
        twobits = False;
        if mode % 2 != 0:
            twobits = True
        print("headersize", headersize, "version", version, "mode", mode, )
        print("treedepth", depth, "coordinate reference system", crs)
        print("Lat/lon of centroid", lat, )
        print("Two bits per voxel:" , twobits)

        # initialise levels
        levels = []
        data = []
        bitcnt = 1

        # pull in the 64 bit chunks and assign to a level.
        for d in range(depth):
            levels.append([])
            newcnt = 0
            for i in range(bitcnt):
                chunk = get_chunk(f)
                newcnt += count_bits(chunk)
                levels[d].append(chunk)
            if twobits:
                data.append([])
                for i in range(bitcnt):
                    chunk = get_chunk(f)
                    newcnt += count_bits(chunk)
                    data[d].append(chunk)
            bitcnt = newcnt

    #test code to make sure it is working!
    vols = []
    for level in levels:
        vols.extend(level)
    print(vols)
    levels = []

    levels.append([get_indexes(vols[0])])
    vols = vols[1:]

    print(levels[0])

    for d in range(1,depth):
        levels.append([])
        for b in range(len(levels[d-1])):
            for i in range(len(levels[d-1][b])):
                levels[d].append(get_indexes(vols[i]))
            vols = vols[len(levels[d-1][b]):]

        print(levels[d])

    l1cnt = 0
    for l0idx, l0val in enumerate(levels[0][0]):
        indices = [l0val]
        for l1idx, l1val in enumerate(levels[1][l0idx]):
            for l2idx, l2val in enumerate(levels[2][l1cnt]):
                print("point index:", l0val, l1val, l2val)
                print("point xyz:", xyz_from_index([l0val, l1val, l2val]))
            l1cnt += 1

def get_chunk(filereader):
    data = filereader.read(8)
    if not data:
        print("prematurely hit end of file")
        exit()
    bit64chunk = struct.unpack('Q', data)[0]
    return bit64chunk

def xyz_from_index(indexes):
    """Generate coordinates from sparse index."""
    x, y, z, = 0, 0, 0
    for level, index in enumerate(indexes):
        mult = pow(4, ((len(indexes) - 1) - level))
        x += (index % 4) * mult
        y += (index % 16 // 4) * mult
        z += (index // 16) * mult
    return (x, y, z)

def count_bits(vol):
    """Count all bits set to 1."""
    count = 0
    vol = np.uint64(vol)
    for i in range(64):
        bit = read_bit(vol, i)
        if bit == 1:
            count += 1
    return count


def get_indexes(vol):
    """Pull the bit indices from a vol."""
    indices = []
    vol = np.uint64(vol)
    for i in range(64):
        bit = read_bit(vol, i)
        if bit == 1:
            indices.append(i)
    return indices


def read_bit(bit64, index):
    """Pull the value at bit index."""
    bit = (int(bit64) >> index) & 1
    return bit


if __name__ == '__main__':
    main()
