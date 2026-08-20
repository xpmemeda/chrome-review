import argparse
import base64
import io
import time
import typing as ty

from flask import Flask, request
from PIL import Image

r"""
Mock server for POST /v1/images/edits with multipart/form-data.

Server arguments:
| Name   | Required | Description                 |
| ------ | -------- | --------------------------- |
| --host | no       | Bind host; defaults to 0.0.0.0. |
| --port | no       | Bind port; defaults to 8512.    |

Request fields:
| Name   | Required | Description                                  |
| ------ | -------- | -------------------------------------------- |
| image  | yes      | Source image file.                           |
| prompt | yes      | Edit instruction.                            |
| mask   | no       | Mask image file; must match the image size.  |
| n      | no       | Output image count; only n=1 is supported.   |

Responses always use b64_json and contain the original image as PNG.
"""

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8512

JsonDict = ty.Dict[str, ty.Any]


def parse_int(values: ty.Mapping[str, ty.Any], name: str, default: int) -> int:
    value = values.get(name)
    return int(value) if value not in (None, "") else default


def parse_str(values: ty.Mapping[str, ty.Any], name: str, default: str) -> str:
    value = values.get(name)
    return str(value) if value not in (None, "") else default


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class MockImageEditServer:
    """Runs a mock image-edit API that returns the uploaded source image."""

    def __init__(self) -> None:
        self.app = Flask(__name__)
        self.app.post("/v1/images/edits")(self.edit_image)

    def edit_image(self) -> ty.Any:
        try:
            image_file = request.files.get("image")
            if image_file is None:
                raise ValueError("missing image")
            image = Image.open(image_file.stream).convert("RGB")

            prompt = parse_str(request.form, "prompt", "")
            if not prompt:
                raise ValueError("missing prompt")

            mask_file = request.files.get("mask")
            if mask_file is not None:
                mask = Image.open(mask_file.stream)
                if mask.size != image.size:
                    raise ValueError(
                        f"mask size {mask.size[0]}x{mask.size[1]} must match "
                        f"image size {image.size[0]}x{image.size[1]}"
                    )

            n = parse_int(request.form, "n", 1)
            if n != 1:
                raise ValueError("only n=1 is supported")
        except Exception as exc:
            return {
                "error": {"message": str(exc), "type": "invalid_request_error"}
            }, 400

        b64_json = base64.b64encode(image_to_png_bytes(image)).decode("ascii")
        image_object: JsonDict = {"b64_json": b64_json, "revised_prompt": prompt}
        return {"created": int(time.time()), "data": [image_object]}

    def run(self, host: str, port: int) -> None:
        self.app.run(host=host, port=port, threaded=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MockImageEditServer().run(args.host, args.port)
