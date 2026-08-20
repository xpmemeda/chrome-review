#!/usr/bin/env python3

import argparse
import base64
import binascii
import io
import json
import sys
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call a flask-diffsynth image-edit endpoint and save its output."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Image-edit endpoint, for example http://127.0.0.1:8512/v1/images/edits.",
    )
    parser.add_argument("--image", required=True, help="Path to the local input image.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path where the edited image will be saved.",
    )
    parser.add_argument(
        "--prompt",
        default="把图片改成卡通风格",
        help="Image-edit instruction.",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    args = parse_args()
    input_path = Path(args.image).expanduser()
    output_path = Path(args.output).expanduser()

    if not input_path.is_file():
        fail(f"input image does not exist or is not a file: {input_path}")
    if args.timeout <= 0:
        fail("--timeout must be positive")

    try:
        with input_path.open("rb") as image_file:
            response = requests.post(
                args.url,
                files={"image": (input_path.name, image_file)},
                data={"prompt": args.prompt, "n": "1"},
                timeout=args.timeout,
            )
    except requests.RequestException as exc:
        fail(f"request failed: {exc}")

    if not response.ok:
        try:
            detail = json.dumps(response.json(), ensure_ascii=False)
        except ValueError:
            detail = response.text[:2000]
        fail(f"server returned HTTP {response.status_code}: {detail}")

    try:
        payload = response.json()
        encoded_image = payload["data"][0]["b64_json"]
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (ValueError, KeyError, IndexError, TypeError, binascii.Error) as exc:
        fail(f"invalid image-edit response: {exc}")

    try:
        with Image.open(io.BytesIO(image_bytes)) as result:
            result.load()
            result_format = result.format or "unknown"
            result_size = result.size
            result_mode = result.mode
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        fail(f"response did not contain a valid image: {exc}")

    print(
        f"OK: HTTP {response.status_code}, received {result_format} "
        f"{result_size[0]}x{result_size[1]} {result_mode}, saved to {output_path}"
    )


if __name__ == "__main__":
    main()
