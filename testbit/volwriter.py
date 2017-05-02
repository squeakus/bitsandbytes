import numpy as np


def set_bit(bit64, index):
     bit64 |= np.uint64(1) << np.uint64(index)
     return np.uint64(bit64)

def unset_bit(bit64, index):
    bit64 &= ~(np.uint64(1) << np.uint64(index))
    return np.uint64(bit64)

def flip_bit(bit64, index):
    bit64 ^= np.uint64(1) << np.uint64(index)
    return np.uint64(bit64)

def read_bit(bit64, index):
    bit = (int(bit64) >> index) & 1
    return bit

def main():
    x = np.uint64(0)
    x = set_bit(x, 63)
    x = set_bit(x, 10)
    print(type(x))
    print("{0:b}".format(x))
    bit = read_bit(x, 63)
    print("63", bit)
    bit = read_bit(x, 62)
    print("62", bit)

if __name__=='__main__':
    main()
