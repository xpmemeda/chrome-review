import argparse
import asyncio
import json
import logging
import time
import typing as ty
import tqdm

import arrival as arrival_lib
import cli
import dataset as dataset_lib
import prefix_hit_profile
from metrics import RequestMetrics, mean, percentile, summarize

JsonDict = ty.Dict[str, ty.Any]
LOG_FORMAT = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
SUCCESS_LEVEL = 25
WARMUP_MAX_ATTEMPTS = 3
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


class ColorFormatter(logging.Formatter):
    COLORS = {
        SUCCESS_LEVEL: "\033[32;1m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[31;1m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.COLORS.get(record.levelno)
        if color is None:
            return message
        return f"{color}{message}{self.RESET}"


def configure_logger(log_path: ty.Optional[str]) -> None:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColorFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    handlers: ty.List[logging.Handler] = [stream_handler]
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        handlers=handlers,
    )
    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def print_summary(summary: JsonDict) -> None:
    logging.info(
        "[%s] ok=%d/%d err=%d elapsed=%.3fs rps=%.3f output_tokens/s=%.1f chunks/s=%.1f",
        summary["label"],
        summary["success"],
        summary["requests"],
        summary["errors"],
        summary["elapsed_sec"],
        summary["rps"],
        summary["output_tokens_per_sec"],
        summary["output_chunks_per_sec"],
    )
    logging.info(
        "[%s] TTFT avg/p50/p90/p99 = %.3f/%.3f/%.3f/%.3fs, "
        "ITL avg/p50/p90/p99 = %.3f/%.3f/%.3f/%.3fs, "
        "E2E avg/p50/p90/p99 = %.3f/%.3f/%.3f/%.3fs, "
        "avg_output_tokens=%.1f avg_output_chars=%.1f",
        summary["label"],
        summary["ttft_avg"],
        summary["ttft_p50"],
        summary["ttft_p90"],
        summary["ttft_p99"],
        summary["itl_avg"],
        summary["itl_p50"],
        summary["itl_p90"],
        summary["itl_p99"],
        summary["e2e_avg"],
        summary["e2e_p50"],
        summary["e2e_p90"],
        summary["e2e_p99"],
        summary["avg_output_tokens"],
        summary["avg_output_chars"],
    )
    for err in summary["first_errors"]:
        logging.info("[%s] sample error: %s", summary["label"], err)


def print_achieved_qps(summary: JsonDict) -> None:
    logging.log(
        SUCCESS_LEVEL,
        "[%s] achieved QPS: %.3f",
        summary["label"],
        summary["rps"],
    )


def render_table(headers: ty.List[str], rows: ty.List[ty.List[str]]) -> str:
    widths = [
        max([len(headers[col_idx])] + [len(row[col_idx]) for row in rows])
        for col_idx in range(len(headers))
    ]

    def render_row(row: ty.List[str]) -> str:
        return "  ".join(
            row[col_idx].rjust(widths[col_idx]) for col_idx in range(len(row))
        )

    lines = [
        render_row(headers),
        render_row(["-" * width for width in widths]),
    ]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def parse_number_list(
    value: ty.Optional[str],
    default_value: ty.Union[int, float],
    cast: ty.Callable[[str], ty.Union[int, float]],
    name: str,
) -> ty.List[ty.Union[int, float]]:
    if not value:
        return [default_value]

    result: ty.List[ty.Union[int, float]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [cast(x) for x in part.split(":")]
            if len(bits) == 2:
                start, stop = bits
                step = 1
            elif len(bits) == 3:
                start, stop, step = bits
            else:
                raise ValueError(f"bad {name} range: {part}")
            x = start
            while x <= stop:
                result.append(x)
                x += step
        else:
            result.append(cast(part))

    seen = set()
    unique = []
    for x in result:
        if x <= 0:
            raise ValueError(f"{name} must be positive")
        if x not in seen:
            seen.add(x)
            unique.append(x)
    return unique


def add_selector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["synthetic-vlm", "synthetic-txt", "jsonl", "omni-multi-message"],
        help=(
            "dataset implementation, synthetic-vlm is text+image, synthetic-txt "
            "is text-only, jsonl loads text messages from a JSONL file, and "
            "omni-multi-message mutates an omni multi-message JSON template"
        ),
        required=True,
    )
    parser.add_argument(
        "--client",
        choices=["mock", "openai", "modelapi", "ark", "diffusion", "ultraman"],
        help="backend client implementation",
        required=True,
    )


def add_common_arguments(
    parser: argparse.ArgumentParser,
    include_concurrency: bool = True,
    include_num_requests: bool = True,
) -> None:
    if include_num_requests:
        parser.add_argument("-n", "--num-requests", type=int, default=100)
    if include_concurrency:
        parser.add_argument("-c", "--concurrency", type=int, default=8)
    parser.add_argument("-w", "--warmup-requests", type=int, default=1)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=None,
        help="Deprecated. tqdm is used for per-request progress.",
    )
    parser.add_argument(
        "--jsonl", help="Append machine-readable summaries to this JSONL file."
    )
    parser.add_argument(
        "-l", "--log-path", help="Optional log file path.", default="benchmark.log"
    )


def add_synthetic_vlm_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--prompt-prefix-hit-rate",
        type=float,
        default=0.0,
        help="Fraction of synthetic prompt tokens shared before the per-request suffix.",
    )
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-seed", type=int, default=None)
    parser.add_argument(
        "--text-seed",
        type=int,
        default=0,
        help="Base seed for text-only synthetic prompts.",
    )
    parser.add_argument(
        "--dataset-path",
        help="Path to a JSONL dataset for -d jsonl.",
    )
    parser.add_argument(
        "--omni-template",
        default="~/workspace/ocean/service_shell/benchmark/omni_multi_message.json",
        help="Path to the omni multi-message JSON template.",
    )
    parser.add_argument(
        "--omni-image-cache",
        default="~/workspace/ocean/service_shell/benchmark/round1_image_base64_cache.json",
        help="Path to URL-to-base64 image cache used by -d omni-multi-message.",
    )
    parser.add_argument(
        "--omni-noise-bytes",
        type=int,
        default=128,
        help="Random bytes appended to the last cached image for each omni request.",
    )
    parser.add_argument(
        "--omni-seed",
        type=int,
        default=0,
        help="Seed for omni per-request mutations.",
    )


def add_client_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "OpenAI base URL for --client openai, full /generate endpoint "
            "for --client diffusion, optional override for --client modelapi/ark, "
            "or host:port/grpc://host:port for --client ultraman."
        ),
    )
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument(
        "--extra-body", help="JSON object merged into request extra_body"
    )
    parser.add_argument(
        "--modelapi-env",
        default="",
        help="Optional x-tt-env header for --client modelapi.",
    )
    parser.add_argument(
        "--diffusion-style",
        help="Fixed style sent to the diffusion service. Defaults to dataset prompt.",
    )
    parser.add_argument(
        "--diffusion-seed",
        type=int,
        help="Base seed sent to the diffusion service. The request index is added.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        help="Inference steps sent as the multipart 'steps' field.",
    )
    parser.add_argument(
        "--diffusion-extra-fields",
        help="JSON object merged into diffusion multipart form fields.",
    )
    parser.add_argument(
        "--ultraman-host",
        default="",
        help="Ultraman gRPC host. Overrides --base-url host if set.",
    )
    parser.add_argument(
        "--ultraman-port",
        type=int,
        default=0,
        help="Ultraman gRPC port. Overrides --base-url port if set.",
    )
    parser.add_argument(
        "--ultraman-proto-path",
        default=cli.ULTRAMAN_PROTO_PATH,
        help="Path containing llmserver.proto.ultraman_pb2 modules. Defaults to bundled proto modules.",
    )
    parser.add_argument(
        "--ultraman-top-k",
        type=int,
        default=1,
        help="top_k sent to ultraman.",
    )
    parser.add_argument(
        "--ultraman-repetition-penalty",
        type=float,
        default=1.1,
        help="repetition_penalty sent to ultraman.",
    )


def build_base_parser(
    include_concurrency: bool = True,
    include_num_requests: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_selector_arguments(parser)
    add_common_arguments(
        parser,
        include_concurrency=include_concurrency,
        include_num_requests=include_num_requests,
    )
    add_synthetic_vlm_dataset_arguments(parser)
    add_client_arguments(parser)
    return parser


def parse_request_count_sweep(value: str, sweep_len: int) -> ty.List[int]:
    result: ty.List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(x) for x in part.split(":")]
            if len(bits) == 2:
                start, stop = bits
                step = 1
            elif len(bits) == 3:
                start, stop, step = bits
            else:
                raise ValueError(f"bad num_requests range: {part}")
            if step <= 0:
                raise ValueError("num_requests range step must be positive")
            x = start
            while x <= stop:
                result.append(x)
                x += step
        else:
            result.append(int(part))

    if not result:
        raise ValueError("--num-requests-sweep must not be empty")
    for x in result:
        if x <= 0:
            raise ValueError("num_requests must be positive")
    if len(result) == 1:
        return result * sweep_len
    if len(result) != sweep_len:
        raise ValueError(
            "--num-requests-sweep length must be 1 or match sweep length: "
            f"{len(result)} != {sweep_len}"
        )
    return result


class BenchmarkEngine:
    def __init__(
        self,
        args: argparse.Namespace,
        total_requests: ty.Optional[int] = None,
    ) -> None:
        self.args = args
        self.dataset = self._build_dataset(total_requests)
        self.detail_log_path = f"{args.log_path}.detail.log" if args.log_path else None

    def build_clients(self, concurrency: int) -> ty.List[ty.Any]:
        if concurrency <= 0:
            raise RuntimeError("--concurrency must be positive")
        if self.args.client == "ultraman":
            compensated_max_tokens = (
                self.args.max_tokens + cli.ULTRAMAN_RESERVED_OUTPUT_TOKENS
            )
            compensated_min_tokens = (
                "N/A"
                if self.args.min_tokens is None
                else str(self.args.min_tokens + cli.ULTRAMAN_RESERVED_OUTPUT_TOKENS)
            )
            logging.warning(
                "Ultraman server reserves %d output tokens; sending max_tokens=%d "
                "and min_tokens=%s after +%d compensation.",
                cli.ULTRAMAN_RESERVED_OUTPUT_TOKENS,
                compensated_max_tokens,
                compensated_min_tokens,
                cli.ULTRAMAN_RESERVED_OUTPUT_TOKENS,
            )
            if self.args.min_tokens is not None:
                logging.warning(
                    "Ultraman gRPC client does not support effective --min-tokens; "
                    "the server may ignore min_tokens/min_new_tokens."
                )
        clients = [self._build_client() for _ in range(concurrency)]
        self._reset_client_metrics(clients)
        return clients

    def _build_dataset(
        self, total_requests: ty.Optional[int]
    ) -> dataset_lib.VlmDataset:
        args = self.args
        if args.dataset == "synthetic-vlm":
            dataset = dataset_lib.SyntheticVlmDataset(
                num_requests=total_requests
                or (args.warmup_requests + args.num_requests),
                num_prompt_tokens=args.num_prompt_tokens,
                prompt_prefix_hit_rate=args.prompt_prefix_hit_rate,
                image_width=args.image_width,
                image_height=args.image_height,
                image_seed=args.image_seed,
            )
        elif args.dataset == "synthetic-txt":
            dataset = dataset_lib.SyntheticTextDataset(
                num_requests=total_requests
                or (args.warmup_requests + args.num_requests),
                num_prompt_tokens=args.num_prompt_tokens,
                prompt_prefix_hit_rate=args.prompt_prefix_hit_rate,
                seed=args.text_seed,
            )
        elif args.dataset == "jsonl":
            if not args.dataset_path:
                raise RuntimeError("--dataset-path is required for -d jsonl")
            dataset = dataset_lib.JsonlTextDataset(dataset_path=args.dataset_path)
        elif args.dataset == "omni-multi-message":
            dataset = dataset_lib.OmniMultiMessageDataset(
                num_requests=total_requests
                or (args.warmup_requests + args.num_requests),
                template_path=args.omni_template,
                image_cache_path=args.omni_image_cache,
                noise_bytes=args.omni_noise_bytes,
                seed=args.omni_seed,
            )
        else:
            raise RuntimeError(f"unknown dataset={args.dataset}")

        if args.client in ("openai", "modelapi", "ark"):
            profiler = prefix_hit_profile.PrefixHitProfiler(dataset)
            message = (
                "dataset profile: "
                f"requests={profiler.request_count} "
                f"num_tokens={profiler.num_tokens} "
                f"hit_tokens={profiler.hit_tokens} "
                f"avg_request_tokens={profiler.avg_request_tokens:.1f} "
                f"avg_hit_tokens={profiler.avg_hit_tokens:.1f} "
                f"token_hit_rate={profiler.token_hit_rate:.4f} "
                f"request_prefix_hit_rate={profiler.request_prefix_hit_rate:.4f}"
            )
            tqdm.tqdm.write(message)
            logging.info(message)
        return dataset

    def _build_client(self) -> ty.Any:
        args = self.args
        if args.client == "mock":
            return cli.MockClient()
        if args.client == "openai":
            if not args.base_url:
                raise RuntimeError("--base-url is required for --client openai")
            if not args.model:
                raise RuntimeError("--model is required for --client openai")
            extra_body = json.loads(args.extra_body) if args.extra_body else None
            return cli.OpenAIClient(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                extra_body=extra_body,
            )
        if args.client == "ark":
            extra_body = json.loads(args.extra_body) if args.extra_body else None
            return cli.ArkClient(
                base_url=args.base_url or cli.ARK_BASE_URL,
                api_key=args.api_key,
                model=args.model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                extra_body=extra_body,
            )
        if args.client == "modelapi":
            extra_body = json.loads(args.extra_body) if args.extra_body else None
            return cli.ModelApiClient(
                base_url=args.base_url or cli.MODELAPI_BASE_URL,
                env=args.modelapi_env,
                model=args.model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                extra_body=extra_body,
            )
        if args.client == "diffusion":
            if not args.base_url:
                raise RuntimeError("--base-url is required for --client diffusion")
            extra_fields = (
                json.loads(args.diffusion_extra_fields)
                if args.diffusion_extra_fields
                else None
            )
            return cli.DiffusionClient(
                url=args.base_url,
                timeout=args.timeout,
                style=args.diffusion_style,
                seed=args.diffusion_seed,
                steps=args.diffusion_steps,
                extra_fields=extra_fields,
            )
        if args.client == "ultraman":
            host = args.ultraman_host
            port = args.ultraman_port
            return cli.UltramanClient(
                host=host,
                port=port,
                proto_path=args.ultraman_proto_path,
                model=args.model,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
                min_tokens=args.min_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.ultraman_top_k,
                repetition_penalty=args.ultraman_repetition_penalty,
            )
        raise RuntimeError(f"unknown client={args.client}")

    async def run_warmup(
        self,
        clients: ty.List[ty.Any],
        num_warmup_requests: int,
        concurrency: int,
        label: str,
        verbose: bool = True,
        request_interval: ty.Optional[float] = None,
    ) -> ty.Optional[JsonDict]:
        if num_warmup_requests < len(clients):
            logging.warning(
                "[%s] warmup requests=%d is smaller than clients=%d; "
                "bumping warmup requests to %d so every client is warmed up",
                label,
                num_warmup_requests,
                len(clients),
                len(clients),
            )
            num_warmup_requests = len(clients)
        if not num_warmup_requests:
            return None
        if verbose:
            if request_interval is None:
                logging.info(
                    "[%s] warmup %d requests, max_attempts=%d",
                    label,
                    num_warmup_requests,
                    WARMUP_MAX_ATTEMPTS,
                )
            else:
                logging.info(
                    "[%s] warmup %d requests with interval=%.3fs, max_attempts=%d",
                    label,
                    num_warmup_requests,
                    request_interval,
                    WARMUP_MAX_ATTEMPTS,
                )

        stime = time.perf_counter()
        detail_label = f"{label}-warmup"
        next_req_idx = 0
        pending_client_indices = [
            req_idx % len(clients) for req_idx in range(num_warmup_requests)
        ]
        last_metrics: ty.List[RequestMetrics] = []

        async def send_one(
            client_idx: int,
            request: dataset_lib.Request,
            semaphore: asyncio.Semaphore,
            req_idx: int,
        ) -> ty.Tuple[int, RequestMetrics]:
            metric = await self._send_prebuilt_one(
                clients[client_idx],
                request,
                semaphore,
                req_idx,
                detail_label,
            )
            return client_idx, metric

        async def run_warmup_attempt(
            attempt: int,
            client_indices: ty.List[int],
            req_idx_base: int,
        ) -> ty.Tuple[ty.List[RequestMetrics], ty.List[int], float]:
            semaphore = asyncio.Semaphore(min(concurrency, len(client_indices)))
            requests = self.dataset.warmup(len(client_indices))
            attempt_start = time.perf_counter()
            tasks = []
            for local_idx, (client_idx, request) in enumerate(
                zip(client_indices, requests)
            ):
                if request_interval is not None:
                    delay = (
                        attempt_start
                        + local_idx * request_interval
                        - time.perf_counter()
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                tasks.append(
                    asyncio.create_task(
                        send_one(client_idx, request, semaphore, req_idx_base + local_idx)
                    )
                )

            metrics: ty.List[RequestMetrics] = []
            failed_client_indices = set()
            for task in tqdm.tqdm(
                asyncio.as_completed(tasks),
                total=len(client_indices),
                desc=f"{label} warmup attempt {attempt}",
                unit="req",
            ):
                client_idx, metric = await task
                metrics.append(metric)
                if not metric.ok:
                    failed_client_indices.add(client_idx)

            return (
                metrics,
                sorted(failed_client_indices),
                time.perf_counter() - attempt_start,
            )

        for attempt in range(1, WARMUP_MAX_ATTEMPTS + 1):
            last_metrics, failed_clients, attempt_elapsed = await run_warmup_attempt(
                attempt,
                pending_client_indices,
                next_req_idx,
            )
            next_req_idx += len(pending_client_indices)
            summary = summarize(last_metrics, attempt_elapsed, f"{label}-warmup")
            if verbose:
                print_summary(summary)
            if not failed_clients:
                if attempt > 1:
                    logging.info(
                        "[%s] warmup recovered after %d attempts",
                        label,
                        attempt,
                    )
                return summary

            if attempt < WARMUP_MAX_ATTEMPTS:
                logging.warning(
                    "[%s] warmup attempt %d failed on clients=%s; retrying",
                    label,
                    attempt,
                    ",".join(str(x) for x in failed_clients),
                )
            pending_client_indices = failed_clients

        elapsed = time.perf_counter() - stime
        summary = summarize(last_metrics, elapsed, f"{label}-warmup")
        if verbose:
            logging.error(
                "[%s] warmup failed after %d attempts; remaining failed clients=%s",
                label,
                WARMUP_MAX_ATTEMPTS,
                ",".join(str(x) for x in pending_client_indices),
            )
        first_errors = "; ".join(summary["first_errors"])
        raise RuntimeError(
            f"[{label}] warmup failed after {WARMUP_MAX_ATTEMPTS} attempts: "
            f"{summary['success']}/{summary['requests']} requests succeeded in "
            f"last attempt. First errors: {first_errors}"
        )

    async def run_closed_loop_once(
        self,
        clients: ty.List[ty.Any],
        num_requests: int,
        concurrency: int,
        progress_interval: ty.Optional[int],
        label: str,
        verbose: bool = True,
        print_summary_at_end: bool = True,
        request_idx_offset: int = 0,
    ) -> JsonDict:
        del progress_interval
        self._reset_client_metrics(clients)
        if verbose:
            logging.info(
                "[%s] start benchmark: requests=%d concurrency=%d clients=%d mode=closed_loop",
                label,
                num_requests,
                concurrency,
                len(clients),
            )
        stime = time.perf_counter()
        metrics = await self._run_closed_loop_work_stealing(
            clients,
            num_requests,
            concurrency,
            request_idx_offset,
            label,
        )
        elapsed = time.perf_counter() - stime
        summary = summarize(metrics, elapsed, label)
        if verbose:
            self._print_client_metrics(label, clients)
        if verbose and print_summary_at_end:
            print_summary(summary)
        if verbose:
            print_achieved_qps(summary)
        return summary

    async def run_open_loop_once(
        self,
        clients: ty.List[ty.Any],
        arrival_plan: ty.List[arrival_lib.ScheduledRequest],
        num_requests: int,
        concurrency: int,
        progress_interval: ty.Optional[int],
        label: str,
        arrival_kind: str,
    ) -> JsonDict:
        del progress_interval
        semaphore = asyncio.Semaphore(concurrency)
        self._reset_client_metrics(clients)
        logging.info(
            "[%s] start benchmark: requests=%d max_concurrency=%d clients=%d arrival=%s",
            label,
            num_requests,
            concurrency,
            len(clients),
            arrival_kind,
        )
        stime = time.perf_counter()
        metrics = await self._run_open_loop_scheduled(
            clients,
            semaphore,
            arrival_plan,
            stime,
            num_requests,
            label,
        )
        elapsed = time.perf_counter() - stime
        summary = summarize(metrics, elapsed, label)
        self._print_client_metrics(label, clients)
        print_summary(summary)
        print_achieved_qps(summary)
        return summary

    def write_jsonl(self, summaries: ty.List[JsonDict]) -> None:
        write_jsonl(self.args.jsonl, summaries)

    @staticmethod
    def log_best_by_rps(summaries: ty.List[JsonDict]) -> None:
        log_best_by_rps(summaries)

    @staticmethod
    def _reset_client_metrics(clients: ty.Iterable[ty.Any]) -> None:
        for client_idx, client in enumerate(clients):
            setattr(client, "_benchmark_client_idx", client_idx)
            setattr(client, "_benchmark_metrics", [])

    @staticmethod
    def _record_client_metric(client: ty.Any, metric: RequestMetrics) -> None:
        metrics = getattr(client, "_benchmark_metrics", None)
        if metrics is None:
            metrics = []
            setattr(client, "_benchmark_metrics", metrics)
        metrics.append(metric)

    @staticmethod
    def _print_client_metrics(label: str, clients: ty.Iterable[ty.Any]) -> None:
        rows = []
        all_metrics = []
        for fallback_idx, client in enumerate(clients):
            client_idx = getattr(client, "_benchmark_client_idx", fallback_idx)
            client_metrics = getattr(client, "_benchmark_metrics", [])
            all_metrics.extend(client_metrics)
            rows.append(
                BenchmarkEngine._client_metrics_row(str(client_idx), client_metrics)
            )

        headers = [
            "client",
            "ok/total",
            "err",
            "ttft_avg",
            "ttft_p50",
            "ttft_p90",
            "e2e_avg",
            "e2e_p50",
            "e2e_p90",
            "tpot",
            "avg_o200k_toks",
            "avg_server_toks",
            "avg_prompt_toks",
            "avg_cached_toks",
            "cache_hit",
            "avg_out_chars",
        ]
        rows.append(BenchmarkEngine._client_metrics_row("all", all_metrics))
        logging.info("[%s] per-client metrics:\n%s", label, render_table(headers, rows))

    @staticmethod
    def _format_optional_mean(values: ty.List[int]) -> str:
        if not values:
            return "N/A"
        return f"{mean(values):.1f}"

    @staticmethod
    def _client_metrics_row(
        name: str, metrics: ty.List[RequestMetrics]
    ) -> ty.List[str]:
        ok_metrics = [metric for metric in metrics if metric.ok]
        ttfts = [metric.ttft for metric in ok_metrics]
        e2es = [metric.e2e for metric in ok_metrics]
        output_tokens = [metric.output_tokens for metric in ok_metrics]
        server_output_tokens = [
            metric.server_output_tokens
            for metric in ok_metrics
            if metric.server_output_tokens is not None
        ]
        server_input_tokens = [
            metric.server_input_tokens
            for metric in ok_metrics
            if metric.server_input_tokens is not None
        ]
        server_cached_tokens = [
            metric.server_cached_tokens
            for metric in ok_metrics
            if metric.server_cached_tokens is not None
        ]
        output_chars = [metric.output_chars for metric in ok_metrics]
        ttft_avg = mean(ttfts)
        e2e_avg = mean(e2es)
        avg_output_tokens = mean(output_tokens)
        avg_server_output_tokens = (
            mean(server_output_tokens) if server_output_tokens else None
        )
        tpot_tokens = avg_server_output_tokens or avg_output_tokens
        tpot = "N/A" if tpot_tokens <= 0 else f"{(e2e_avg - ttft_avg) / tpot_tokens:.4f}"
        cache_hit = BenchmarkEngine._format_cache_hit_rate(
            server_cached_tokens,
            server_input_tokens,
        )
        return [
            name,
            f"{len(ok_metrics)}/{len(metrics)}",
            str(len(metrics) - len(ok_metrics)),
            f"{ttft_avg:.3f}",
            f"{percentile(ttfts, 50):.3f}",
            f"{percentile(ttfts, 90):.3f}",
            f"{e2e_avg:.3f}",
            f"{percentile(e2es, 50):.3f}",
            f"{percentile(e2es, 90):.3f}",
            tpot,
            f"{avg_output_tokens:.1f}",
            BenchmarkEngine._format_optional_mean(server_output_tokens),
            BenchmarkEngine._format_optional_mean(server_input_tokens),
            BenchmarkEngine._format_optional_mean(server_cached_tokens),
            cache_hit,
            f"{mean(output_chars):.1f}",
        ]

    @staticmethod
    def _format_cache_hit_rate(
        cached_tokens: ty.List[int],
        input_tokens: ty.List[int],
    ) -> str:
        if not cached_tokens or not input_tokens:
            return "N/A"
        total_input_tokens = sum(input_tokens)
        if total_input_tokens <= 0:
            return "N/A"
        return f"{sum(cached_tokens) / total_input_tokens:.4f}"

    def _write_detail_log(
        self,
        label: str,
        client: ty.Any,
        metric: RequestMetrics,
    ) -> None:
        if not self.detail_log_path:
            return
        item = {
            "label": label,
            "client_idx": getattr(client, "_benchmark_client_idx", None),
            "req_idx": metric.req_idx,
            "ok": metric.ok,
            "ttft": metric.ttft,
            "e2e": metric.e2e,
            "output_tokens_o200k": metric.output_tokens,
            "server_output_tokens": metric.server_output_tokens,
            "server_input_tokens": metric.server_input_tokens,
            "server_cached_tokens": metric.server_cached_tokens,
            "server_usage": metric.server_usage,
            "server_raw_chunks": metric.server_raw_chunks,
            "output_chars": metric.output_chars,
            "output_chunks": metric.output_chunks,
            "error": metric.error,
            "output": metric.output_text,
        }
        with open(self.detail_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    async def _send_dataset_one(
        self,
        client: ty.Any,
        semaphore: asyncio.Semaphore,
        req_idx: int,
        label: str,
    ) -> RequestMetrics:
        async with semaphore:
            return await self._send_request(
                client,
                self.dataset.get(req_idx),
                req_idx,
                label,
            )

    async def _send_prebuilt_one(
        self,
        client: ty.Any,
        request: dataset_lib.Request,
        semaphore: asyncio.Semaphore,
        req_idx: int,
        label: str,
    ) -> RequestMetrics:
        async with semaphore:
            return await self._send_request(client, request, req_idx, label)

    async def _send_request(
        self,
        client: ty.Any,
        request: dataset_lib.Request,
        req_idx: int,
        label: str,
    ) -> RequestMetrics:
        metric = await client.send_request(req_idx, request)
        self._write_detail_log(label, client, metric)
        metric_for_summary = metric._replace(output_text="")
        self._record_client_metric(client, metric_for_summary)
        return metric_for_summary

    async def _run_closed_loop_work_stealing(
        self,
        clients: ty.List[ty.Any],
        num_requests: int,
        concurrency: int,
        request_idx_offset: int,
        label: str,
    ) -> ty.List[RequestMetrics]:
        request_queue: asyncio.Queue[int] = asyncio.Queue()
        results: asyncio.Queue[RequestMetrics] = asyncio.Queue()
        for idx in range(num_requests):
            request_queue.put_nowait(request_idx_offset + idx)

        async def worker(client: ty.Any) -> None:
            while True:
                try:
                    req_idx = request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    metric = await self._send_request(
                        client, self.dataset.get(req_idx), req_idx, label
                    )
                except Exception as e:
                    metric = RequestMetrics(
                        req_idx, False, 0.0, 0.0, 0, 0, 0, error=repr(e)
                    )
                    self._write_detail_log(label, client, metric)
                    self._record_client_metric(client, metric)
                await results.put(metric)

        worker_count = min(concurrency, num_requests)
        worker_tasks = [
            asyncio.create_task(worker(clients[idx % len(clients)]))
            for idx in range(worker_count)
        ]

        metrics: ty.List[RequestMetrics] = []
        progress = tqdm.tqdm(
            total=num_requests,
            desc=f"{label} benchmark",
            unit="req",
        )
        try:
            for _ in range(num_requests):
                metric = await results.get()
                metrics.append(metric)
                progress.update(1)
        finally:
            progress.close()

        await asyncio.gather(*worker_tasks)
        return metrics

    async def _run_open_loop_scheduled(
        self,
        clients: ty.List[ty.Any],
        semaphore: asyncio.Semaphore,
        arrival_plan: ty.List[arrival_lib.ScheduledRequest],
        start_at: float,
        num_requests: int,
        label: str,
    ) -> ty.List[RequestMetrics]:
        results: asyncio.Queue[ty.Union[RequestMetrics, BaseException]] = (
            asyncio.Queue()
        )
        request_tasks: ty.List[asyncio.Task[None]] = []

        async def tracked_send(
            client: ty.Any,
            scheduled_request: arrival_lib.ScheduledRequest,
        ) -> None:
            try:
                metric = await self._send_dataset_one(
                    client,
                    semaphore,
                    scheduled_request.req_idx,
                    label,
                )
            except Exception as e:
                await results.put(e)
                raise
            await results.put(metric)

        async def launch_scheduled_requests() -> None:
            for scheduled_request in arrival_plan:
                delay = start_at + scheduled_request.scheduled_at - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                client = clients[scheduled_request.req_idx % len(clients)]
                request_tasks.append(
                    asyncio.create_task(tracked_send(client, scheduled_request))
                )
                launch_progress.update(1)

        metrics: ty.List[RequestMetrics] = []
        launch_progress = tqdm.tqdm(
            total=num_requests,
            desc=f"{label} launch",
            unit="req",
            position=0,
        )
        complete_progress = tqdm.tqdm(
            total=num_requests,
            desc=f"{label} complete",
            unit="req",
            position=1,
        )
        launcher_task = asyncio.create_task(launch_scheduled_requests())

        async def monitor_launcher() -> None:
            try:
                await launcher_task
            except Exception as e:
                await results.put(e)
                raise

        launcher_monitor = asyncio.create_task(monitor_launcher())
        try:
            while len(metrics) < num_requests:
                item = await results.get()
                if isinstance(item, BaseException):
                    raise item
                metrics.append(item)
                complete_progress.update(1)
        finally:
            launch_progress.close()
            complete_progress.close()
            if len(metrics) < num_requests:
                for task in request_tasks + [launcher_task, launcher_monitor]:
                    task.cancel()
                await asyncio.gather(
                    *request_tasks,
                    launcher_task,
                    launcher_monitor,
                    return_exceptions=True,
                )

        await launcher_monitor
        await asyncio.gather(*request_tasks)
        return metrics


def write_jsonl(path: ty.Optional[str], summaries: ty.List[JsonDict]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        for summary in summaries:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def log_best_by_rps(summaries: ty.List[JsonDict]) -> None:
    if len(summaries) <= 1:
        return
    best = max(summaries, key=lambda x: x["rps"])
    logging.info(
        "best by RPS: %s rps=%.3f e2e_p99=%.3fs ttft_p99=%.3fs",
        best["label"],
        best["rps"],
        best["e2e_p99"],
        best["ttft_p99"],
    )
