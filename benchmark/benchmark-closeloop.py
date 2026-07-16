#!/usr/bin/env python3
import asyncio
import logging
import shlex
import sys

import cmdargs
import engine


def build_parser():
    parser = cmdargs.build_base_parser(
        include_concurrency=False,
        include_num_requests=False,
    )
    parser.add_argument(
        "--concurrency-sweep",
        type=int,
        nargs="+",
        required=True,
        help="one or more global in-flight concurrency values, e.g. 1 2 4 8 16",
    )
    parser.add_argument(
        "--num-requests-sweep",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Requests per concurrency point. Values must match "
            "--concurrency-sweep length."
        ),
    )
    return parser


async def async_main(args):
    concurrencies = args.concurrency_sweep
    request_counts = args.num_requests_sweep
    if len(request_counts) != len(concurrencies):
        raise ValueError(
            "--num-requests-sweep length must match --concurrency-sweep length: "
            f"{len(request_counts)} != {len(concurrencies)}"
        )
    if any(num_requests <= 0 for num_requests in request_counts):
        raise ValueError("--num-requests-sweep values must be positive")
    total_requests = sum(request_counts)
    benchmark = engine.BenchmarkEngine(args, total_requests=total_requests)
    summaries = []
    request_idx_offset = 0
    for concurrency, num_requests in zip(concurrencies, request_counts):
        clients = benchmark.build_clients(concurrency)
        label = f"c{concurrency}-closed_loop"
        await benchmark.run_warmup(
            clients,
            args.warmup_requests,
            concurrency,
            label,
            verbose=True,
        )
        summaries.append(
            await benchmark.run_closed_loop_once(
                clients,
                num_requests,
                concurrency,
                args.progress_interval,
                label,
                verbose=True,
                print_summary_at_end=False,
                request_idx_offset=request_idx_offset,
            )
        )
        request_idx_offset += num_requests

    benchmark.write_jsonl(summaries)
    print_result_table(summaries)


def print_result_table(summaries):
    rows = []
    for summary in summaries:
        concurrency = summary["label"].split("-", 1)[0].removeprefix("c")
        rows.append(
            [
                concurrency,
                f"{summary['rps']:.3f}",
                f"{summary['success_rate'] * 100.0:.2f}%",
                f"{summary['success']}/{summary['requests']}",
            ]
        )

    headers = ["concurrency", "qps", "success_rate", "success/total"]
    logging.info("closed-loop sweep results:\n%s", engine.render_table(headers, rows))


def main():
    parser = build_parser()
    args = parser.parse_args()
    engine.configure_logger(args.log_path)
    logging.info("command: %s", shlex.join(sys.argv))
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
