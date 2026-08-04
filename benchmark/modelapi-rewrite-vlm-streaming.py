"""
- 首轮有没有图片: 有
- 会不会删图: 会，最多保留 5 张图片，第 6 轮开始删图。
- 失败处理: 重试直至成功。
"""

import argparse
import asyncio
import base64
import json
import logging
import multiprocessing
import random
import shlex
import sys
import typing as ty

import cli
import dataset as dataset_lib
import engine

JsonDict = ty.Dict[str, ty.Any]
IMAGE_WIDTH = 632
IMAGE_HEIGHT = 1400
IMAGE_TARGET_KIB = 32
MAX_HISTORY_IMAGES = 5
PROMPT_FILLER = "hi"


class RandomImageGenerator:
    def __init__(
        self,
        width: int = IMAGE_WIDTH,
        height: int = IMAGE_HEIGHT,
        target_kib: int = IMAGE_TARGET_KIB,
    ) -> None:
        self.width = width
        self.height = height
        self.target_bytes = target_kib * dataset_lib.BYTES_PER_KIB

    def get(self, seed: int, round_idx: int) -> str:
        image_seed = seed * 1000003 + round_idx
        rows = dataset_lib.make_base_rgb_rows(self.width, self.height, image_seed)
        self._patch_rows(rows, image_seed)
        png = dataset_lib.encode_png_rgb(self.width, self.height, rows)
        if len(png) > self.target_bytes:
            raise RuntimeError(
                f"generated PNG is larger than target size: {len(png)} > "
                f"{self.target_bytes}"
            )
        png = dataset_lib.pad_png_to_size(png, self.target_bytes, image_seed)
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")

    def _patch_rows(self, rows: bytearray, image_seed: int) -> None:
        rng = random.Random(image_seed)
        num_pixels = min(64, self.width * self.height)
        for _ in range(num_pixels):
            pos = rng.randrange(self.width * self.height)
            y, x = divmod(pos, self.width)
            offset = y * (1 + self.width * 3) + 1 + x * 3
            rows[offset] = rng.randrange(256)
            rows[offset + 1] = rng.randrange(256)
            rows[offset + 2] = rng.randrange(256)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Temporary ModelApi rewrite-history VLM chat CLI.",
        allow_abbrev=False,
    )
    parser.add_argument("--base-url", default=cli.MODELAPI_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--modelapi-env", default="")
    parser.add_argument(
        "--process",
        type=int,
        default=1,
        help="Number of OS processes to run.",
    )
    parser.add_argument(
        "-c",
        "--concurrency-per-process",
        "--concurrence-per-process",
        dest="concurrency_per_process",
        type=int,
        default=1,
        help="Number of concurrent ModelApi clients in each process.",
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
            "User message metadata template. Supports {round}, {seed}, "
            "{client}, and {iteration}."
        ),
    )
    parser.add_argument(
        "--num-prompt-tokens",
        type=int,
        default=8000,
        help=(
            "Total text tokens generated for the first round, excluding prompt "
            "metadata."
        ),
    )
    parser.add_argument(
        "--prompt-prefix-hit-rate",
        type=float,
        default=0.5,
        help="Fraction of first-round text tokens placed in the shared system prefix.",
    )
    parser.add_argument(
        "--rewrite-tokens",
        "--hi-per-round",
        dest="rewrite_tokens",
        type=int,
        default=4000,
        help="Number of 'hi' tokens in the rewritten tail user text.",
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
        default="modelapi-rewrite-vlm-streaming.log",
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
    suffix_tokens: int,
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
    if suffix_tokens <= 0:
        return prompt
    return prompt + " " + " ".join([PROMPT_FILLER] * suffix_tokens)


def make_system_prompt(prefix_tokens: int) -> str:
    if prefix_tokens <= 0:
        return ""
    return " ".join([PROMPT_FILLER] * prefix_tokens)


def make_image_message(image_url: str) -> JsonDict:
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
        ],
    }


def make_text_message(prompt: str) -> JsonDict:
    return {"role": "user", "content": prompt}


def append_jsonl(path: ty.Optional[str], record: JsonDict) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_once(
    args: argparse.Namespace,
    modelapi_client: cli.ModelApiClient,
    image_generator: RandomImageGenerator,
    client_idx: int,
    iteration_idx: int,
    seed: int,
) -> None:
    prefix_tokens = int(args.num_prompt_tokens * args.prompt_prefix_hit_rate)
    suffix_tokens = args.num_prompt_tokens - prefix_tokens
    fixed_user_tokens = suffix_tokens - args.rewrite_tokens
    base_messages: ty.List[JsonDict] = []
    system_prompt = make_system_prompt(prefix_tokens)
    if system_prompt:
        base_messages.append({"role": "system", "content": system_prompt})
    fixed_user_prompt = format_prompt(
        args.prompt,
        0,
        fixed_user_tokens,
        client_idx,
        iteration_idx,
        seed,
    )
    base_messages.append({"role": "user", "content": fixed_user_prompt})
    image_messages: ty.List[JsonDict] = []

    for round_idx in range(args.rounds):
        tail_prompt = format_prompt(
            args.prompt,
            round_idx,
            args.rewrite_tokens,
            client_idx,
            iteration_idx,
            seed,
        )
        image_messages.append(make_image_message(image_generator.get(seed, round_idx)))
        pruned_images = max(0, len(image_messages) - MAX_HISTORY_IMAGES)
        if pruned_images:
            image_messages = image_messages[pruned_images:]
        tail_text_message = make_text_message(tail_prompt)
        request_messages = base_messages + image_messages + [tail_text_message]
        request = {"messages": request_messages}

        attempt = 0
        while True:
            attempt += 1
            logging.info(
                "[client %d iteration %d seed %d round %d/%d attempt %d] "
                "sending rewrite request, messages=%d images=%d pruned_images=%d",
                client_idx,
                iteration_idx,
                seed,
                round_idx + 1,
                args.rounds,
                attempt,
                len(request_messages),
                len(image_messages),
                pruned_images,
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
                    "images": len(image_messages),
                    "pruned_images": pruned_images,
                    "messages": request_messages,
                    "error": metric.error,
                    "e2e": metric.e2e,
                },
            )
            await asyncio.sleep(1.0)

        append_jsonl(
            args.jsonl,
            {
                "client": client_idx,
                "iteration": iteration_idx,
                "seed": seed,
                "round": round_idx + 1,
                "ok": True,
                "images": len(image_messages),
                "pruned_images": pruned_images,
                "messages": request_messages,
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


async def worker(
    args: argparse.Namespace,
    process_idx: int,
    local_client_idx: int,
) -> None:
    modelapi_client = build_client(args)
    image_generator = RandomImageGenerator()
    client_idx = process_idx * args.concurrency_per_process + local_client_idx
    total_clients = args.process * args.concurrency_per_process
    for iteration_idx in range(args.iterations):
        seed = args.seed + client_idx + iteration_idx * total_clients
        modelapi_client.user = (
            f"benchmark-modelapi-process{process_idx}-client{client_idx}-seed{seed}"
        )
        logging.info(
            "[process %d client %d iteration %d seed %d] start",
            process_idx,
            client_idx,
            iteration_idx,
            seed,
        )
        await run_once(
            args,
            modelapi_client,
            image_generator,
            client_idx,
            iteration_idx,
            seed,
        )
        logging.info(
            "[process %d client %d iteration %d seed %d] finished",
            process_idx,
            client_idx,
            iteration_idx,
            seed,
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.process <= 0:
        raise RuntimeError("--process must be positive")
    if args.concurrency_per_process <= 0:
        raise RuntimeError("--concurrency-per-process must be positive")
    if args.rounds <= 0:
        raise RuntimeError("--rounds must be positive")
    if args.iterations <= 0:
        raise RuntimeError("--iterations must be positive")
    if args.num_prompt_tokens <= 0:
        raise RuntimeError("--num-prompt-tokens must be positive")
    if args.rewrite_tokens < 0:
        raise RuntimeError("--rewrite-tokens must be non-negative")
    if args.prompt_prefix_hit_rate < 0.0 or args.prompt_prefix_hit_rate > 1.0:
        raise RuntimeError("--prompt-prefix-hit-rate must be in [0, 1]")
    prefix_tokens = int(args.num_prompt_tokens * args.prompt_prefix_hit_rate)
    suffix_tokens = args.num_prompt_tokens - prefix_tokens
    if args.rewrite_tokens > suffix_tokens:
        raise RuntimeError(
            "--rewrite-tokens must be no larger than the first-round suffix tokens"
        )
    if args.sleep_seconds < 0.0:
        raise RuntimeError("--sleep-seconds must be non-negative")
    if args.sleep_after_first_round_seconds < 0.0:
        raise RuntimeError("--sleep-after-first-round-seconds must be non-negative")


async def async_main(args: argparse.Namespace, process_idx: int) -> None:
    tasks = [
        asyncio.create_task(worker(args, process_idx, local_client_idx))
        for local_client_idx in range(args.concurrency_per_process)
    ]
    await asyncio.gather(*tasks)


def run_process(args: argparse.Namespace, process_idx: int) -> None:
    asyncio.run(async_main(args, process_idx))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    engine.configure_logger(args.log_path)
    logging.info("command: %s", shlex.join(sys.argv))
    logging.info(
        "starting rewrite VLM streaming test: processes=%d "
        "concurrency_per_process=%d total_clients=%d",
        args.process,
        args.concurrency_per_process,
        args.process * args.concurrency_per_process,
    )
    if args.process == 1:
        run_process(args, 0)
        return

    processes = [
        multiprocessing.Process(
            target=run_process,
            args=(args, process_idx),
        )
        for process_idx in range(args.process)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed_processes = [process.pid for process in processes if process.exitcode != 0]
    if failed_processes:
        raise RuntimeError(f"child processes failed: {failed_processes}")


if __name__ == "__main__":
    main()
