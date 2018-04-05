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
        # If using twobits then extract that too!
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
                    data[d].append(chunk)
            bitcnt = newcnt

    coordinates = get_coordinates(levels, data, depth)

def get_coordinates(levels, data, depth):
    coordinates = []
    indexes = get_all_indexes(levels, data, depth)
    indexes = traverse_indexes([], indexes, depth-1, [0] * depth)
    for index in indexes:
        coordinates.append(xyz_from_index(index))
    print(coordinates)

def get_all_indexes(levels, data, depth):
    bitcnt = 1
    indexes = []

    for d in range(depth):
        newcnt = 0
        indexes.append([])
        for b in range(bitcnt):
            chunk_indexes = get_indexes(levels[d][b])
            newcnt += len(chunk_indexes)
            indexes[d].append(chunk_indexes)
        bitcnt = newcnt

    for levidx in indexes:
        print(levidx)
    return indexes

def traverse_indexes(prev, levels, depth, levelcnt):
    #we are doing a depth first traversal but need to keep track of
    # our BFS position because the breadth position is required!
    traversed = []
    if depth > 0:
        block = levels[0][levelcnt[depth]]
        for index in block:
            lowerlist = prev + [index]
            result = traverse_indexes(lowerlist, levels[1:], depth-1, levelcnt)
            traversed.extend(result)
    else:
        block = levels[0][levelcnt[depth]]
        for index in block:
            traversed.append(prev + [index])
    levelcnt[depth] += 1
    return traversed


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


class IndexTree(object):
    def __init__(self):
        self.children = None
        self.data = None

if __name__ == '__main__':
    main()
