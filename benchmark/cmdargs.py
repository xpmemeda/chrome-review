import argparse
import cli


def build_base_parser(
    include_concurrency: bool = True,
    include_num_requests: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)

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

    parser.add_argument("--synthetic-txt-num-prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--synthetic-txt-prompt-prefix-hit-rate",
        type=float,
        default=0.0,
        help=(
            "Fraction of synthetic text prompt tokens shared before the "
            "per-request suffix."
        ),
    )
    parser.add_argument("--synthetic-vlm-num-prompt-tokens", type=int, default=None)
    parser.add_argument(
        "--synthetic-vlm-prompt-prefix-hit-rate",
        type=float,
        default=0.0,
        help=(
            "Fraction of synthetic VLM prompt tokens shared before the "
            "per-request suffix."
        ),
    )
    parser.add_argument("--synthetic-vlm-image-width", type=int, default=None)
    parser.add_argument("--synthetic-vlm-image-height", type=int, default=None)
    parser.add_argument("--synthetic-vlm-image-seed", type=int, default=None)
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
        "--omni-seed",
        type=int,
        default=0,
        help="Seed for omni per-request mutations.",
    )

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
    return parser
