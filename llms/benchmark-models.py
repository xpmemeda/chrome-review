import os
import re
import subprocess
import collections
import argparse

log = "benchmark-async-all-models.log"


def run_benchmark_script(
    backend,
    model,
    num_coroutine,
    num_prompt_tokens,
    num_prompt_hit_tokens,
    num_output_tokens,
):
    num_prompts = num_coroutine * (16 if num_output_tokens >= 128 else 32)

    env = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), f"activate-{backend}"
    )
    benchmark_script = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "benchmark-async.py"
    )
    command = (
        f"source {env} && "
        f"python {benchmark_script} "
        f"--log {log} "
        f"--backend {backend} "
        f"--model /home/wnr/llms/{model} "
        f"--n {num_prompts} "
        f"--b {num_coroutine} "
        f"--i {num_prompt_tokens} "
        f"--h {num_prompt_hit_tokens} "
        f"--o {num_output_tokens} "
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
    )
    stdout, stderr = process.communicate()
    print("-" * 50)
    print(command)
    print(stdout)
    print(stderr)


def benchmark_all_models(backends, models, num_coroutines, ihos):
    for backend in backends:
        for model in models:
            for num_coroutine in num_coroutines:
                for i, h, o in ihos:
                    run_benchmark_script(backend, model, num_coroutine, i, h, o)


def parse_log():
    with open(log, "r") as f:
        text = f.read()

    Key = collections.namedtuple(
        "key",
        [
            "cpu",
            "gpu",
            "backend",
            "version",
            "model",
            "b",  # num_coroutines
            "i",  # num_prompt_tokens
            "h",  # num_prompt_hit_tokens
            "o",  # num_output_tokens
        ],
    )
    benchmark_results = collections.OrderedDict()

    pattern = (
        r"""Benchmark start: """
        r"""device=Device\(CPU='(?P<cpu>.*?)', GPU='(?P<gpu>.*?)'\), """
        r"""backend=(?P<backend>.*?)-(?P<version>.*?), model=/home/wnr/llms/(?P<model>.*?), """
        r"""num_coroutine=(?P<b>.*?), num_prompt_tokens=(?P<i>.*?), num_prompt_hit_tokens=(?P<h>.*?), num_output_tokens=(?P<o>.*?)"""
        r"""\n"""
    )
    for match in re.finditer(pattern=pattern, string=text):
        benchmark_results[
            Key(
                match.group("cpu"),
                match.group("gpu"),
                match.group("backend"),
                match.group("version"),
                match.group("model"),
                int(match.group("b")),
                int(match.group("i")),
                int(match.group("h")),
                int(match.group("o")),
            )
        ] = "e"

    pattern = (
        r"""Benchmark done: """
        r"""device=Device\(CPU='(?P<cpu>.*?)', GPU='(?P<gpu>.*?)'\), """
        r"""backend=(?P<backend>.*?)-(?P<version>.*?), model=/home/wnr/llms/(?P<model>.*?), """
        r"""num_coroutine=(?P<b>.*?), num_prompt_tokens=(?P<i>.*?), num_prompt_hit_tokens=(?P<h>.*?), num_output_tokens=(?P<o>.*?)"""
        r""", """
        r"""Metrics=MetricsManager\(Num prompts .*? Avg TTFT (?P<ttft>.*?) Avg E2E (?P<e2e>.*?) RPS (?P<rps>.*?) TPS (?P<tps>.*?)\)"""
    )

    for match in re.finditer(pattern, text):
        benchmark_results[
            Key(
                match.group("cpu"),
                match.group("gpu"),
                match.group("backend"),
                match.group("version"),
                match.group("model"),
                int(match.group("b")),
                int(match.group("i")),
                int(match.group("h")),
                int(match.group("o")),
            )
        ] = match

    hdr = "|Framework  |Version    |CPU;GPU    |B/I/H/O        |Avg TTFT (s)   |Avg E2E (s)    |RPS    |TPS        |"
    sep = "|-          |-          |-          |-              |-              |-              |-      |-          |"
    col_widths = [len(x) for x in hdr.split("|")[1:-1]]

    def device_map(name):
        mapdata = {
            "AMD EPYC 7K62 48-Core Processor": "7K62",
            "AMD EPYC 7K83 64-Core Processor": "7K83",
            "AMD EPYC 9K84 96-Core Processor": "9K84",
            "NVIDIA A10": "A10",
            "NVIDIA A100-SXM4-40GB": "A100",
            "NVIDIA L20": "L20",
            "NVIDIA L40": "L40",
        }
        if name in mapdata:
            return mapdata[name]

        return name

    def serialize_match(key, match):
        backend = key.backend
        version = key.version
        cpu = key.cpu
        gpu = key.gpu
        b, i, h, o = key.b, key.i, key.h, key.o
        biho = f"""{b}/{i}/{h}/{o}"""

        row_items = [backend, version, f"{device_map(cpu)};{device_map(gpu)}", biho]

        if match == "e":
            row_items += ["e", "e", "e", "e"]
        elif match == "m":
            row_items += ["m", "m", "m", "m"]
        else:
            assert isinstance(match, re.Match)
            row_items += [
                match.group("ttft"),
                match.group("e2e"),
                match.group("rps"),
                match.group("tps"),
            ]

        row = "|".join(f"{item:<{col_widths[i]}}" for i, item in enumerate(row_items))
        return "|" + row + "|"

    def model_sort_key(model):
        order = {"Qwen2-0.5B": 1, "Qwen2-7B": 2}
        return 0 if model not in order else order[model]

    def backend_sort_key(backend: tuple):
        order = {"trtllm": 1, "sglang": 2, "vllm": 3, "wnr": 4}
        return 0 if backend[0] not in order else order[backend[0]]

    devices = set()
    backends = set()
    models = set()
    bihos = set()
    for key in benchmark_results.keys():
        devices.add((key.cpu, key.gpu))
        models.add(key.model)
        backends.add((key.backend, key.version))
        bihos.add((key.b, key.i, key.h, key.o))

    for model in sorted(models, key=model_sort_key):
        tables = f"\n## {model}\n"
        for cpu, gpu in devices:
            rows = []
            for b, i, h, o in sorted(bihos):
                for backend, version in sorted(backends, key=backend_sort_key):
                    key = Key(cpu, gpu, backend, version, model, b, i, h, o)
                    match = (
                        "m" if key not in benchmark_results else benchmark_results[key]
                    )
                    rows.append(serialize_match(key, match))

            table = "\n".join([hdr, sep] + rows)
            table = "\n" + table + "\n"
            tables += table

        print(tables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do-benchmark", action="store_true")
    parser.add_argument("--log", type=str, required=True)
    arguments = parser.parse_args()

    do_benchmark = arguments.do_benchmark
    log = arguments.log

    backends = ["trtllm", "vllm", "sglang", "wnr"]
    models = ["Qwen2-0.5B", "Qwen2-7B"]
    num_coroutines = [1, 4, 8]
    ihos = [
        (1024, 0, 32),
        (1024, 512, 32),
        (1024, 1024, 32),
        (32, 0, 1024),
        (512, 0, 512),
    ]

    if do_benchmark:
        benchmark_all_models(backends, models, num_coroutines, ihos)

    parse_log()
