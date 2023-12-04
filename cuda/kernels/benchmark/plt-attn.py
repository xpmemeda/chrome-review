import matplotlib.pyplot as plt
import re
import os
import glob
import sys
import argparse
import collections


def get_txt_files(directory):
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    return txt_files


def get_label_and_mean(path) -> dict:
    with open(path, "r") as f:
        text = f.read()
    lines = text.split("\n")
    label_and_mean = {}
    pattern = r"(?P<key>.*?):.*?mean=(?P<mean>\d+\.\d+)"
    for line in lines:
        match = re.search(pattern, line)
        if match:
            label_and_mean[match.group("key")] = float(match.group("mean"))
    return label_and_mean


def parse_file_name(path) -> dict:
    file_name = os.path.basename(path)
    pattern = r"([a-zA-Z]+)(\d+)"
    matches = re.findall(pattern, file_name)
    result = {key: int(value) for key, value in matches}
    return result


def is_dict_a_contains_dict_b(a, b):
    for key, value in b.items():
        if key not in a or a[key] != value:
            return False
    return True


def plt_model(model, nq, nk, d):
    directory = sys.argv[1]
    txt_files = get_txt_files(directory)

    for b, marker in zip([4, 8, 16], ["o", "x", "d"]):
        target = {"b": b, "nq": nq, "nk": nk, "d": d}
        x = {}
        for file in txt_files:
            xx = parse_file_name(file)
            if is_dict_a_contains_dict_b(xx, target):
                x[xx["s"]] = file
        od = collections.OrderedDict(sorted(x.items()))
        s = []
        v1_1 = []
        v1_8 = []
        for k, v in od.items():
            s.append(k)
            xxxx = get_label_and_mean(v)
            if "paged_attention_v1_1" in xxxx:
                v1_1.append(xxxx["paged_attention_v1_1"])
            if "paged_attention_v1_8" in xxxx:
                v1_8.append(xxxx["paged_attention_v1_8"])

        plt.plot(s, v1_1, color="red", label="b%d.vllm" % b, marker=marker)
        plt.plot(s, v1_8, color="blue", label="b%d.wnr" % b, marker=marker)
    plt.legend()
    plt.xlabel("s")
    plt.ylabel("latency(ms)")
    plt.title(f"{model}: nq={nq} nk={nk} d={d}")
    plt.savefig(f"{model}.png")
    plt.clf()


if __name__ == "__main__":
    plt_model("Hunyuan-0.5B", 10, 10, 64)
    plt_model("Qwen2-0.5B", 14, 2, 64)
    plt_model("Qwen2-7B", 28, 4, 128)
    plt_model("Baichuan2-7B", 32, 32, 128)
