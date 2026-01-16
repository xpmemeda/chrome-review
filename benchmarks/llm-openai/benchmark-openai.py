import time
import argparse
from functools import partial
import logging
import typing as ty

from suite.datasets import get_cls as datasets_get_cls
from suite.cli import get_cls as cli_get_cls
from suite.tokenizer import Tokenizer
from suite.metrics import MetricsRecorder
from suite.limiter import (
    NoopConcurrencyGate,
    NoopPacer,
    QpsPacer,
    SemaphoreConcurrencyGate,
)
from suite.orchestrator import CoOrch, MtOrch, MpOrch


def configure_logger(fpath: ty.Optional[str] = None):
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


def main(cmd_arguments):
    configure_logger(cmd_arguments.log_path)

    # check arguments.
    if not cmd_arguments.qps and not cmd_arguments.max_batch:
        raise RuntimeError("qps OR max_batch should at least specify one.")

    tokenizer = Tokenizer(cmd_arguments.tokenizer)

    # datasets.
    dataset = datasets_get_cls(cmd_arguments.dataset).from_args(cmd_arguments)

    # metrics
    metrics_recorder = MetricsRecorder(tokenizer)

    # pace and concurrency gate.
    if cmd_arguments.qps:
        pacer = QpsPacer(cmd_arguments.qps, cmd_arguments.num_requests)
    else:
        pacer = NoopPacer()

    if cmd_arguments.max_batch:
        concurrency_gate = SemaphoreConcurrencyGate(cmd_arguments.max_batch)
    else:
        concurrency_gate = NoopConcurrencyGate()

    start_at = time.time() + 5.0

    # orch.
    client_factory = partial(cli_get_cls(cmd_arguments.client).from_args, cmd_arguments)
    if cmd_arguments.orch == "mp":
        orch_cls = MpOrch
    elif cmd_arguments.orch == "mt":
        orch_cls = MtOrch
    else:
        orch_cls = CoOrch
    logging.warning(f"benchmark will start at 5 seconds later, please wait...")
    orch = orch_cls(
        client_factory=client_factory,
        n=cmd_arguments.num_requests,
        start_at=start_at,
        dataset=dataset,
        tokenizer=tokenizer,
        pacer=pacer,
        concurrency_gate=concurrency_gate,
        metrics_recorder=metrics_recorder,
    )

    end_at = orch.run()
    elapsed = end_at - start_at

    logging.warning(
        f"Process {cmd_arguments.num_requests} requests use {elapsed : .2f} seconds, archived qps {cmd_arguments.num_requests / elapsed :.2f}"
    )

    logging.warning(metrics_recorder.report_table())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=["synthetic", "jsonl"],
        help="dataset type.",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--client",
        type=str,
        choices=["mock", "openai", "spam"],
        help="backend client implementation.",
        required=True,
    )
    parser.add_argument(
        "--orch",
        type=str,
        choices=["co", "mt", "mp"],
        default="co",
        help="client manager impl, coroutine or multi-thread or multi-process",
    )
    parser.add_argument(
        "-l",
        "--log-path",
        type=str,
        help="result log path.",
        default="benchmark.log",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        help="tokenizer path, used to calculate num tokens.",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        help="number of prompts to send.",
        required=True,
    )
    parser.add_argument(
        "-w",
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
        "--max-batch",
        type=int,
        help="the max number of processes working on send requests. this may cause qps can not reach the specified value.",
    )
    preview_arguments, _ = parser.parse_known_args()
    cli_get_cls(preview_arguments.client).add_arguments(parser)
    datasets_get_cls(preview_arguments.dataset).add_arguments(parser)

    cmd_arguments = parser.parse_args()
    main(cmd_arguments)
