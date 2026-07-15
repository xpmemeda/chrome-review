#!/usr/bin/env python3
import argparse
import asyncio
import logging
import shlex
import sys

import arrival as arrival_lib
import engine


def build_arrival_planner(args, qps):
    if qps <= 0:
        raise RuntimeError("--qps-sweep values must be positive")
    if args.arrival == "poisson":
        return arrival_lib.PoissonPlanner(qps, args.arrival_seed)
    if args.arrival == "constant":
        return arrival_lib.ConstantRatePlanner(qps)
    raise RuntimeError(f"unknown arrival={args.arrival}")


def build_parser():
    parser = engine.build_base_parser(include_num_requests=False)
    parser.add_argument(
        "--arrival",
        choices=["poisson", "constant"],
        help="global request arrival process",
        required=True,
    )
    parser.add_argument(
        "--qps-sweep",
        required=True,
        help="comma list/ranges, e.g. 8 or 1,2,4,8 or 1:8:1",
    )
    parser.add_argument(
        "--num-requests-sweep",
        required=True,
        help=(
            "Requests per QPS point. A single value is broadcast to all points; "
            "lists/ranges must match --qps-sweep length."
        ),
    )
    parser.add_argument("--arrival-seed", type=int, default=0)
    return parser


def log_arrival_plan(label, target_qps, arrival_plan):
    if len(arrival_plan) <= 1:
        logging.info(
            "[%s] arrival plan: target_qps=%.6g requests=%d scheduled_qps=N/A",
            label,
            target_qps,
            len(arrival_plan),
        )
        return

    first_scheduled_at = arrival_plan[0].scheduled_at
    last_scheduled_at = arrival_plan[-1].scheduled_at
    scheduled_window_sec = last_scheduled_at - first_scheduled_at
    scheduled_qps = (
        (len(arrival_plan) - 1) / scheduled_window_sec
        if scheduled_window_sec > 0.0
        else float("inf")
    )
    logging.info(
        "[%s] arrival plan: target_qps=%.6g scheduled_qps=%.6g "
        "requests=%d scheduled_window=%.3fs first_at=%.3fs last_at=%.3fs",
        label,
        target_qps,
        scheduled_qps,
        len(arrival_plan),
        scheduled_window_sec,
        first_scheduled_at,
        last_scheduled_at,
    )


async def async_main(args: argparse.Namespace):
    qps_values = [
        float(x) for x in engine.parse_number_list(args.qps_sweep, 1.0, float, "qps")
    ]
    request_counts = engine.parse_request_count_sweep(
        args.num_requests_sweep,
        len(qps_values),
    )

    total_requests = sum(request_counts)
    benchmark = engine.BenchmarkEngine(args, total_requests=total_requests)
    summaries = []
    request_idx_offset = 0
    for qps, num_requests in zip(qps_values, request_counts):
        clients = benchmark.build_clients(args.concurrency)
        label = f"qps{qps:g}-{args.arrival}"
        arrival_plan = [
            arrival_lib.ScheduledRequest(
                req_idx=scheduled_request.req_idx + request_idx_offset,
                scheduled_at=scheduled_request.scheduled_at,
            )
            for scheduled_request in build_arrival_planner(args, qps).plan(num_requests)
        ]
        log_arrival_plan(label, qps, arrival_plan)
        await benchmark.run_warmup(
            clients,
            args.warmup_requests,
            args.concurrency,
            label,
            request_interval=1.0 / qps,
        )
        summaries.append(
            await benchmark.run_open_loop_once(
                clients,
                arrival_plan,
                num_requests,
                args.concurrency,
                args.progress_interval,
                label,
                args.arrival,
            )
        )
        request_idx_offset += num_requests

    benchmark.write_jsonl(summaries)
    benchmark.log_best_by_rps(summaries)


def main():
    parser = build_parser()
    args = parser.parse_args()
    engine.configure_logger(args.log_path)
    logging.info("command: %s", shlex.join(sys.argv))
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
