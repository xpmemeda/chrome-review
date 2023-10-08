import argparse
import re
import statistics


def extract_prefix_float(string):
    nums = re.findall("\d+\.\d+", string)
    assert len(nums) == 1, string
    return float(nums[0])


def analys_nsys(file, primary_index, index):
    items = {}
    with open(file, "r") as f:
        line = f.readline()
        col_num = len(line.rstrip("\n").split("\t"))
        line = f.readline()
        while line:
            c = line.rstrip("\n").split("\t")
            key = c[primary_index]
            if key not in items:
                items[key] = []
            value = c[index]
            items[key].append(extract_prefix_float(value))

            line = f.readline()

    items = {k: sum(v) for k, v in items.items()}
    items = {
        k: v for k, v in sorted(items.items(), key=lambda item: item[1], reverse=True)
    }
    return items


def main(args):
    ns1 = analys_nsys(args.file[0], args.primary_index, args.index)
    ns2 = analys_nsys(args.file[1], args.primary_index, args.index)

    ns1_sum = 0
    ns2_sum = 0
    common_keys = [k for k in ns1.keys() if k in ns2.keys()]
    for k in common_keys:
        print("\033[92m" + k + "\033[0m")
        v1 = ns1[k] if k in ns1 else 0
        v2 = ns2[k] if k in ns2 else 0
        print(v1, v2)
        ns1_sum += v1
        ns2_sum += v2
    ns1_keys = [k for k in ns1.keys() if k not in ns2.keys()]
    for k in ns1_keys:
        print("\033[93m" + k + "\033[0m")
        v1 = ns1[k] if k in ns1 else 0
        v2 = ns2[k] if k in ns2 else 0
        print(v1, v2)
        ns1_sum += v1
        ns2_sum += v2
    ns2_keys = [k for k in ns2.keys() if k not in ns1.keys()]
    for k in ns2_keys:
        print("\033[94m" + k + "\033[0m")
        v1 = ns1[k] if k in ns1 else 0
        v2 = ns2[k] if k in ns2 else 0
        print(v1, v2)
        ns1_sum += v1
        ns2_sum += v2

    print(ns1_sum, ns2_sum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", nargs="+", type=str)
    parser.add_argument("--primary-index", type=int, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    main(args)
