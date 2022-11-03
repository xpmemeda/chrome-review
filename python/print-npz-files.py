import numpy as np
import sys


def print_npz_files():
    data = np.load(sys.argv[1])
    for arr_name in data.files:
        arr = data[arr_name]
        print("{}: dtype={}, shape={}".format(arr_name, arr.dtype, arr.shape))


if __name__ == "__main__":
    print_npz_files()
