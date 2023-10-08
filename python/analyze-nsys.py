import argparse
import re
import statistics


def extract_value_and_unit(string):
    nums = re.findall("\d+\.\d+", string)
    unit = re.findall("μs|ms", string)
    assert len(nums) == 1, string
    assert len(unit) == 1, string
    return float(nums[0]), unit[0]


def analys_nsys(file, index):
    r"""
    Sum the value of specific index in a nsys ``Events View``.

    Args:
        file: str, the txt file contenting ``Events View``.
        index: int, the index to read value and sum it.

    Returns:
        (1): int, the sum value, unit μs.
        (2): int, total line num.
    """
    units_map = {"μs": 1, "ms": 1000}
    sum_v = 0
    num_line = 0
    with open(file, "r") as f:
        line = f.readline()
        c = line.rstrip("\n").split("\t")
        print("name = %s" % c[index])
        line = f.readline()
        while line:
            c = line.rstrip("\n").split("\t")
            value, unit = extract_value_and_unit(c[index])
            sum_v += value * units_map[unit]
            num_line += 1
            line = f.readline()
    return sum_v, num_line


def main(args):
    sum_v, num_line = analys_nsys(args.file, args.index)
    print(sum_v, num_line)
    print(sum_v / num_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--index", type=int, required=True, help="sum of column[index]")
    args = parser.parse_args()
    main(args)
