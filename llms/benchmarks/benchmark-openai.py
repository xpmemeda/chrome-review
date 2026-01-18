import sys
import inspect
import time
import random
import numpy
import openai
import argparse
import logging
import concurrent.futures
import datasets

from typing import Union, Optional, List
from tokenizer import Tokenizer
from metrics import Metric, Metrics
from datasetprofiler import DatasetProfiler


def configure_logger(fpath: Optional[str] = None):
    handlers = [logging.StreamHandler()]
    if fpath:
        handlers.append(logging.FileHandler(fpath))
    format = "%(asctime)s.%(msecs)03d - %(process)d - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    logging.basicConfig(
        level=getattr(logging, "WARNING"),
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=handlers,
    )


def send_request(
    cid: str,
    host: str,
    port: int,
    api_key: str,
    model: str,
    messages: dict,
    completions_args: dict,
):
    try:

        client = openai.OpenAI(
            base_url=f"http://{host}:{port}/v1", api_key=api_key, max_retries=0
        )

        stime = time.time()
        ttft_time = None
        ttfs_time = None
        itl_base = stime
        itl_list = []

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **completions_args,
        )

        resp_text = ""
        for chunk in completion:
            ttft_time = ttft_time or time.time()
            itl_time = time.time()
            itl_list.append(itl_time - itl_base)
            itl_base = itl_time

            resp = chunk.choices[0].delta.content
            if resp:
                resp_text += resp

                ttfs_time = ttfs_time or time.time()

        etime = time.time()

        ttft = ttft_time - stime
        ttfs = ttfs_time - stime
        e2e = etime - stime
        itl_list.pop(0)

        return Metric(cid, ttft, ttfs, itl_list, e2e, messages, resp_text)

    except Exception as e:
        logging.error(f"send_request error, {cid=}. exception:\n{e}")

        raise RuntimeError("send_request error.")


randint_generator = None


def deal_sampling_params(sampling_params, cmd_arguments, cid: str):
    global randint_generator
    if not randint_generator:
        randint_generator = random.Random(0)

    sampling_params["temperature"] = cmd_arguments.temperature

    if cmd_arguments.max_tokens and cmd_arguments.min_tokens:
        num_tokens = randint_generator.randint(
            cmd_arguments.min_tokens, cmd_arguments.max_tokens
        )
        sampling_params["max_tokens"] = num_tokens
        sampling_params["min_tokens"] = num_tokens
    elif cmd_arguments.max_tokens:
        if "max_tokens" in sampling_params:
            logging.error(
                "dataset has it's max_tokens, cmd_arguments.max_tokens will be ignored."
            )
        else:
            sampling_params["max_tokens"] = cmd_arguments.max_tokens
    elif cmd_arguments.min_tokens:
        if "min_tokens" in sampling_params:
            logging.error(
                "dataset has it's min_tokens, cmd_arguments.min_tokens will be ignored."
            )
        else:
            sampling_params["min_tokens"] = cmd_arguments.min_tokens

    if cmd_arguments.cid:
        sampling_params["cid"] = cid

    if cmd_arguments.c:
        sampling_params["internal_debug_app_id"] = cmd_arguments.c

    completions_args = sampling_params

    if "extra_headers" not in completions_args:
        completions_args["extra_headers"] = {}
    if "extra_body" not in completions_args:
        completions_args["extra_body"] = {}

    # sgl.
    if "min_tokens" in completions_args:
        min_tokens = completions_args.pop("min_tokens")
        completions_args["extra_body"]["min_tokens"] = min_tokens

    # apex.
    if "cid" in completions_args:
        cid = completions_args.pop("cid")
        completions_args["extra_body"]["client_id"] = cid
    else:
        cid = None

    if "internal_debug_app_id" in completions_args:
        appid = completions_args.pop("internal_debug_app_id")
        completions_args["extra_headers"]["internal_debug_app_id"] = appid

    return completions_args


def main(cmd_arguments):
    configure_logger(cmd_arguments.o)

    if not cmd_arguments.qps and not cmd_arguments.max_batch:
        raise RuntimeError("qps AND max_batch should at least specify one.")
    if cmd_arguments.qps and cmd_arguments.max_batch:
        raise RuntimeError("qps AND max_batch should can't be both set.")

    tokenizer = Tokenizer(cmd_arguments.t)
    dataset: datasets.Dataset = getattr(datasets, cmd_arguments.d)(
        dataset_path=cmd_arguments.f
    )
    logging.warning("load dataset done.")

    dataset_profiler = DatasetProfiler(
        dataset, tokenizer, cmd_arguments.dataset_profiler_radix_size
    )
    logging.warning(f"DatasetProfiler:\n\t{dataset_profiler.report()}")

    metrics = Metrics(tokenizer)

    max_num_workers = 0
    num_workers = 0
    num_requests = 0

    def cb(future):
        nonlocal num_workers
        num_workers -= 1

        try:
            metrics.update(future.result())
        except:
            logging.error(f"Req error.")

    host = cmd_arguments.host
    port = cmd_arguments.port
    api_key = cmd_arguments.api_key
    model = cmd_arguments.model

    if dataset.sleep_seconds() and cmd_arguments.qps:
        raise RuntimeError("dataset has its qps, please check.")
    elif cmd_arguments.qps:
        sleep_seconds = numpy.random.exponential(scale=1.0 / cmd_arguments.qps, size=cmd_arguments.n).tolist()
    elif dataset.sleep_seconds():
        sleep_seconds = dataset.sleep_seconds()
    else:
        sleep_seconds = 0.0

    # JIT warmup.
    for _ in range(cmd_arguments.num_warmup_requests):
        metric = send_request(
            None,
            host,
            port,
            api_key,
            model,
            messages=[{"role": "user", "content": "hello "}],
            completions_args={"max_tokens": 4},
        )
        logging.warning(f"warmup: {metric=}")

    t0 = time.time()

    if cmd_arguments.max_batch:
        init_sleeps = [0.5] * cmd_arguments.max_batch
    else:
        init_sleeps = []

    with concurrent.futures.ProcessPoolExecutor() as executor:

        last_worker_launch_time = time.time()

        while request := dataset.get():

            if num_requests >= cmd_arguments.n:
                break

            cid = f"{cmd_arguments.cid}-{num_requests}" if cmd_arguments.cid else None

            messages, sampling_params = request
            sampling_params = deal_sampling_params(sampling_params, cmd_arguments, cid)

            future = executor.submit(
                send_request,
                cid,
                host,
                port,
                api_key,
                model,
                messages,
                sampling_params,
            )
            future.add_done_callback(cb)

            num_workers += 1
            num_requests += 1
            max_num_workers = max(max_num_workers, num_workers)

            logging.warning(
                f"Add a worker, {num_workers=}, {num_requests=}, sleeped_seconds={time.time() - last_worker_launch_time}"
            )
            last_worker_launch_time = time.time()

            # sleep.
            if cmd_arguments.max_batch:
                while num_workers >= cmd_arguments.max_batch:
                    time.sleep(0.01)
            if isinstance(sleep_seconds, list):
                time.sleep(sleep_seconds[num_requests - 1])
            else:
                time.sleep(sleep_seconds)
            if init_sleeps:
                time.sleep(init_sleeps.pop())

        while num_workers:
            logging.warning(f"{num_workers=}, waiting...")
            time.sleep(1)

    t1 = time.time()

    logging.warning(f"{metrics}")
    logging.warning(metrics.wesearch_metrics())
    logging.warning(f"{max_num_workers=}")
    logging.warning(f"qps={num_requests / (t1 - t0)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d",
        type=str,
        choices=[
            k
            for k, v in datasets.__dict__.items()
            if inspect.isclass(v) and issubclass(v, datasets.Dataset)
        ],
        required=True,
    )
    parser.add_argument(
        "--f", type=str, help="dataset path, set EMPTY if --d Random", required=True
    )
    parser.add_argument(
        "--dataset-profiler-radix-size",
        type=int,
        default=sys.maxsize,
        help="The radix tree size to profiler prefix hit radio.",
    )
    parser.add_argument("--o", type=str, help="result log path.", default="")
    parser.add_argument(
        "--t",
        type=str,
        help="tokenizer path, used to calculate num tokens.",
        required=True,
    )
    parser.add_argument(
        "--n",
        type=int,
        help="number of prompts to send.",
        required=True,
    )
    parser.add_argument(
        "--c",
        type=str,
        help="target cluster, use EMPTY if not benchmarking with conductor.",
        required=True,
    )
    parser.add_argument(
        "--num-warmup-requests",
        type=int,
        help="number requests to warmup.",
        required=True,
    )
    parser.add_argument(
        "--qps",
        type=float,
        help="target qps to benchmark. 0 means using the dataset qps.",
    )
    parser.add_argument(
        "--qps-adjust-method",
        type=str,
        choices=["NONE", "sample", "stretch"],
        help="the method to adjust dataset's qps. it was required when dataset's qps not equal to specified qps(--qps). set to NONE to use dataset's qps.",
        default="NONE",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        help="the max number of processes working on send requests. this may cause qps can not reach the specified value.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        help="the min tokens to generate, 0 means not set.",
        default=0,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="the max tokens to generate, 0 means not set.",
        default=0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="sampling params.",
        default=1.0,
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="insert a random str before prompt to avoid prefix cache.",
    )
    parser.add_argument(
        "--cid",
        type=str,
        help="add client id to debug svr log.",
    )
    parser.add_argument("--host", type=str, help="openai server host.", required=True)
    parser.add_argument("--port", type=int, help="openai server port.", required=True)
    parser.add_argument(
        "--model", type=str, help="for example: deepseek-ai/DeepSeek-R1", required=True
    )
    parser.add_argument("--api-key", type=str, default="dummy")
    cmd_arguments = parser.parse_args()
    main(cmd_arguments)
