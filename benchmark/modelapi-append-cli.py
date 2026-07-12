#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import shlex
import sys
import typing as ty

import cli
import dataset as dataset_lib
import engine

JsonDict = ty.Dict[str, ty.Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Temporary ModelApi append-history chat CLI.",
        allow_abbrev=False,
    )
    parser.add_argument("--base-url", default=cli.MODELAPI_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--modelapi-env", default="")
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent ModelApi clients.",
    )
    parser.add_argument("--rounds", type=int, default=15, help="Conversation rounds.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of full multi-round tests each client runs.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument(
        "--prompt",
        default="seed={seed} client={client} iteration={iteration} round={round}，继续。",
        help=(
            "User message prefix appended each round. Supports {round}, {seed}, "
            "{client}, and {iteration}."
        ),
    )
    parser.add_argument(
        "--hi-per-round",
        "--append-hi-tokens",
        dest="hi_per_round",
        type=int,
        default=4000,
        help="Number of 'hi' tokens appended to each round's user message.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Seconds to sleep between rounds.",
    )
    parser.add_argument(
        "--sleep-after-first-round-seconds",
        type=float,
        default=0.0,
        help="Seconds to sleep after round 1 only.",
    )
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--extra-body", help="JSON object merged into extra_body.")
    parser.add_argument(
        "--jsonl",
        help="Optional path to append per-round request and response records.",
    )
    parser.add_argument(
        "-l",
        "--log-path",
        help="Optional log file path.",
        default="modelapi-append-cli.log",
    )
    return parser


def build_client(args: argparse.Namespace) -> cli.ModelApiClient:
    extra_body = json.loads(args.extra_body) if args.extra_body else None
    return cli.ModelApiClient(
        base_url=args.base_url,
        env=args.modelapi_env,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        extra_body=extra_body,
    )


def format_prompt(
    template: str,
    round_idx: int,
    hi_per_round: int,
    client_idx: int,
    iteration_idx: int,
    seed: int,
) -> str:
    prompt = template.format(
        round=round_idx + 1,
        client=client_idx,
        iteration=iteration_idx,
        seed=seed,
    )
    if hi_per_round <= 0:
        return prompt
    return prompt + " " + " ".join(["hi"] * hi_per_round)


def append_jsonl(path: ty.Optional[str], record: JsonDict) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_once(
    args: argparse.Namespace,
    modelapi_client: cli.ModelApiClient,
    client_idx: int,
    iteration_idx: int,
    seed: int,
) -> None:
    messages: ty.List[JsonDict] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    for round_idx in range(args.rounds):
        user_message = {
            "role": "user",
            "content": format_prompt(
                args.prompt,
                round_idx,
                args.hi_per_round,
                client_idx,
                iteration_idx,
                seed,
            ),
        }
        messages.append(user_message)
        request = {"messages": list(messages)}

        attempt = 0
        while True:
            attempt += 1
            logging.info(
                "[client %d iteration %d seed %d round %d/%d attempt %d] "
                "sending request, messages=%d",
                client_idx,
                iteration_idx,
                seed,
                round_idx + 1,
                args.rounds,
                attempt,
                len(messages),
            )
            metric = await modelapi_client.send_request(round_idx, request)
            if metric.ok:
                break
            logging.error(
                "[client %d iteration %d seed %d round %d/%d attempt %d] "
                "request failed, retrying: %s",
                client_idx,
                iteration_idx,
                seed,
                round_idx + 1,
                args.rounds,
                attempt,
                metric.error,
            )
            append_jsonl(
                args.jsonl,
                {
                    "client": client_idx,
                    "iteration": iteration_idx,
                    "seed": seed,
                    "round": round_idx + 1,
                    "attempt": attempt,
                    "ok": False,
                    "messages": messages,
                    "error": metric.error,
                    "e2e": metric.e2e,
                },
            )
            await asyncio.sleep(1.0)

        assistant_message = {"role": "assistant", "content": metric.output_text}
        messages.append(assistant_message)
        append_jsonl(
            args.jsonl,
            {
                "client": client_idx,
                "iteration": iteration_idx,
                "seed": seed,
                "round": round_idx + 1,
                "ok": True,
                "messages": messages,
                "assistant": metric.output_text,
                "ttft": metric.ttft,
                "e2e": metric.e2e,
                "output_tokens": metric.output_tokens,
                "server_output_tokens": metric.server_output_tokens,
                "server_input_tokens": metric.server_input_tokens,
                "server_cached_tokens": metric.server_cached_tokens,
                "server_usage": metric.server_usage,
                "server_raw_chunks": metric.server_raw_chunks,
            },
        )
        logging.info(
            "[client %d iteration %d seed %d round %d/%d] done, "
            "output_chars=%d output_tokens=%d e2e=%.3fs",
            client_idx,
            iteration_idx,
            seed,
            round_idx + 1,
            args.rounds,
            metric.output_chars,
            metric.output_tokens,
            metric.e2e,
        )
        print(
            f"\n===== client {client_idx} iteration {iteration_idx} "
            f"seed {seed} round {round_idx + 1} assistant ====="
        )
        print(metric.output_text)
        if round_idx == 0 and args.sleep_after_first_round_seconds > 0.0:
            logging.info(
                "[client %d iteration %d seed %d round 1/%d] sleeping %.3fs "
                "after first round",
                client_idx,
                iteration_idx,
                seed,
                args.rounds,
                args.sleep_after_first_round_seconds,
            )
            await asyncio.sleep(args.sleep_after_first_round_seconds)
        if round_idx + 1 < args.rounds and args.sleep_seconds > 0.0:
            logging.info(
                "[client %d iteration %d seed %d round %d/%d] sleeping %.3fs",
                client_idx,
                iteration_idx,
                seed,
                round_idx + 1,
                args.rounds,
                args.sleep_seconds,
            )
            await asyncio.sleep(args.sleep_seconds)


async def worker(args: argparse.Namespace, client_idx: int) -> None:
    modelapi_client = build_client(args)
    for iteration_idx in range(args.iterations):
        seed = args.seed + client_idx + iteration_idx * args.concurrency
        modelapi_client.user = f"benchmark-modelapi-client{client_idx}-seed{seed}"
        logging.info(
            "[client %d iteration %d seed %d] start",
            client_idx,
            iteration_idx,
            seed,
        )
        await run_once(args, modelapi_client, client_idx, iteration_idx, seed)
        logging.info(
            "[client %d iteration %d seed %d] finished",
            client_idx,
            iteration_idx,
            seed,
        )


async def async_main(args: argparse.Namespace) -> None:
    if args.concurrency <= 0:
        raise RuntimeError("--concurrency must be positive")
    if args.rounds <= 0:
        raise RuntimeError("--rounds must be positive")
    if args.iterations <= 0:
        raise RuntimeError("--iterations must be positive")
    if args.sleep_seconds < 0.0:
        raise RuntimeError("--sleep-seconds must be non-negative")
    if args.sleep_after_first_round_seconds < 0.0:
        raise RuntimeError("--sleep-after-first-round-seconds must be non-negative")

    tasks = [
        asyncio.create_task(worker(args, client_idx))
        for client_idx in range(args.concurrency)
    ]
    await asyncio.gather(*tasks)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    engine.configure_logger(args.log_path)
    logging.info("command: %s", shlex.join(sys.argv))
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
