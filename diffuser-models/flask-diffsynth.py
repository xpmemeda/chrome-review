import argparse
import base64
import binascii
import io
import logging
import os
import queue
import threading
import time
import typing as ty
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field

import torch
from diffsynth.pipelines.flux2_image import (
    Flux2ImagePipeline,
    ModelConfig as Flux2ModelConfig,
)
from diffsynth.pipelines.qwen_image import (
    ModelConfig as QwenModelConfig,
    QwenImagePipeline,
)
from flask import Flask, request, send_file
from PIL import Image

PIPELINE_KIND_QWEN = "qwen"
PIPELINE_KIND_FLUX2 = "flux2"
HOST = "0.0.0.0"
PORT = 8512

MODEL = ""
QWEN_BASE_MODEL = "Qwen/Qwen-Image"
QWEN_PROCESSOR_MODEL = "Qwen/Qwen-Image-Edit"
PIPELINE_KIND = PIPELINE_KIND_QWEN
ENABLE_CFG = False
DEFAULT_STEPS = 0
DEFAULT_CFG_SCALE = 4.0
DEFAULT_VARIATION_PROMPT = (
    "Generate a variation of the provided image. Return only the generated image, "
    "not a text description."
)


app = Flask(__name__)
work_queue: "queue.Queue[WorkItem]" = queue.Queue()


@dataclass(frozen=True)
class WorkKey:
    steps: int
    cfg_scale: float
    negative_prompt: str
    height: ty.Optional[int]
    width: ty.Optional[int]


@dataclass
class WorkItem:
    image: Image.Image
    prompt: str
    seed: int
    key: WorkKey
    done: threading.Event = field(default_factory=threading.Event)
    result: ty.Optional[bytes] = None
    error: ty.Optional[BaseException] = None


JsonDict = ty.Dict[str, ty.Any]
FormValues = ty.Mapping[str, ty.Any]
Pipeline = ty.Any


def parse_int(values: FormValues, name: str, default: int) -> int:
    value = values.get(name)
    return int(value) if value not in (None, "") else default


def parse_optional_int(values: FormValues, name: str) -> ty.Optional[int]:
    value = values.get(name)
    return int(value) if value not in (None, "") else None


def parse_float_alias(
    values: FormValues, names: ty.Sequence[str], default: float
) -> float:
    for name in names:
        value = values.get(name)
        if value not in (None, ""):
            return float(value)
    return default


def parse_str(values: FormValues, name: str, default: str) -> str:
    value = values.get(name)
    return str(value) if value not in (None, "") else default


def decode_data_url(url: str) -> bytes:
    prefix, sep, payload = url.partition(",")
    if sep != "," or ";base64" not in prefix:
        raise ValueError("image_url must be a base64 data URL")
    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid image_url base64 payload") from exc


def read_image_url(url: str) -> bytes:
    if url.startswith("data:"):
        return decode_data_url(url)
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def image_from_json_value(value: ty.Any) -> Image.Image:
    if isinstance(value, str):
        return image_from_bytes(read_image_url(value))
    if isinstance(value, dict) and isinstance(value.get("url"), str):
        return image_from_bytes(read_image_url(value["url"]))
    raise ValueError("image_url must be a string or an object with url")


def extract_json_prompt(value: ty.Any) -> str:
    parts: ty.List[str] = []
    if isinstance(value, str):
        parts.append(value)
    elif isinstance(value, list):
        for item in value:
            parts.extend(extract_json_prompt(item).splitlines())
    elif isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            parts.append(text)
        content = value.get("content")
        if content is not None:
            parts.extend(extract_json_prompt(content).splitlines())
    return "\n".join(part for part in parts if part)


def extract_json_image(value: ty.Any) -> ty.Optional[Image.Image]:
    if isinstance(value, dict):
        if value.get("type") in ("input_image", "image_url"):
            image_url = value.get("image_url") or value.get("url")
            if image_url is not None:
                return image_from_json_value(image_url)
        for key in ("image_url", "image"):
            if key in value:
                return image_from_json_value(value[key])
        images = value.get("images")
        if isinstance(images, list) and images:
            return image_from_json_value(images[0])
        for key in ("input", "content", "messages"):
            image = extract_json_image(value.get(key))
            if image is not None:
                return image
    elif isinstance(value, list):
        for item in value:
            image = extract_json_image(item)
            if image is not None:
                return image
    return None


def image_from_form() -> Image.Image:
    for name in ("image", "image[]", "images", "images[]"):
        files = request.files.getlist(name)
        if files:
            return Image.open(files[0].stream).convert("RGB")
    if request.form.get("image_url"):
        return image_from_json_value(request.form["image_url"])
    raise ValueError("missing image")


def build_item_from_values(
    image: Image.Image, values: FormValues, default_prompt: str
) -> WorkItem:
    prompt = parse_str(values, "prompt", default_prompt)
    seed = parse_int(values, "seed", 0)
    steps = parse_int(values, "steps", DEFAULT_STEPS)
    cfg_scale = parse_float_alias(
        values, ("cfg_scale", "guidance_scale"), DEFAULT_CFG_SCALE
    )
    negative_prompt = parse_str(values, "negative_prompt", "")

    return WorkItem(
        image=image,
        prompt=prompt,
        seed=seed,
        key=WorkKey(
            steps=steps,
            cfg_scale=cfg_scale,
            negative_prompt=negative_prompt,
            height=parse_optional_int(values, "height"),
            width=parse_optional_int(values, "width"),
        ),
    )


def build_generate_item() -> WorkItem:
    image = image_from_form()
    style = request.form.get("style") or "摄影后期"
    prompt = (
        request.form.get("prompt") or f"请对图片进行{style}风格的编辑，保持主体一致。"
    )
    return build_item_from_values(image, request.form, prompt)


def build_edit_item_from_form() -> WorkItem:
    image = image_from_form()
    default_prompt = request.form.get("style") or DEFAULT_VARIATION_PROMPT
    return build_item_from_values(image, request.form, default_prompt)


def build_edit_item_from_json(body: JsonDict) -> WorkItem:
    image = extract_json_image(body)
    if image is None:
        raise ValueError("missing image")

    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        prompt = extract_json_prompt(body.get("input")) or extract_json_prompt(
            body.get("messages")
        )
    values = dict(body)
    values["prompt"] = prompt or DEFAULT_VARIATION_PROMPT
    return build_item_from_values(image, values, DEFAULT_VARIATION_PROMPT)


def enqueue_and_wait(item: WorkItem) -> bytes:
    work_queue.put(item)
    item.done.wait()

    if item.error is not None:
        raise item.error
    if item.result is None:
        raise RuntimeError("empty generation result")
    return item.result


@app.post("/generate")
def generate():
    try:
        item = build_generate_item()
    except Exception as exc:
        return {"error": str(exc)}, 400

    try:
        result = enqueue_and_wait(item)
    except Exception as exc:
        logging.error("generation failed: %r", exc)
        return {"error": repr(exc)}, 500

    return send_file(io.BytesIO(result), mimetype="image/png")


@app.post("/v1/images/edits")
def edit_image():
    try:
        if request.is_json:
            body = request.get_json(silent=True)
            if not isinstance(body, dict):
                raise ValueError("JSON request body must be an object")
            item = build_edit_item_from_json(body)
            values: FormValues = body
        else:
            item = build_edit_item_from_form()
            values = request.form
        n = parse_int(values, "n", 1)
        if n != 1:
            raise ValueError("only n=1 is supported")
    except Exception as exc:
        return {"error": {"message": str(exc), "type": "invalid_request_error"}}, 400

    try:
        result = enqueue_and_wait(item)
    except Exception as exc:
        logging.error("image edit failed: %r", exc)
        return {"error": {"message": repr(exc), "type": "server_error"}}, 500

    b64_json = base64.b64encode(result).decode("ascii")
    response_format = parse_str(values, "response_format", "b64_json")
    image_object: JsonDict = {"revised_prompt": item.prompt}
    if response_format == "url":
        image_object["url"] = f"data:image/png;base64,{b64_json}"
    elif response_format in ("b64_json", ""):
        image_object["b64_json"] = b64_json
    else:
        return {
            "error": {
                "message": "response_format must be 'b64_json' or 'url'",
                "type": "invalid_request_error",
            }
        }, 400

    return {"created": int(time.time()), "data": [image_object]}


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def infer_pipeline_kind(model: str) -> str:
    normalized = model.lower()
    if "flux2" in normalized or "flux.2" in normalized or "flux_2" in normalized:
        return PIPELINE_KIND_FLUX2
    return PIPELINE_KIND_QWEN


def load_flux2_pipeline() -> Pipeline:
    if os.path.isdir(MODEL):
        model_configs = [
            Flux2ModelConfig(
                path=[
                    str(path)
                    for path in sorted(
                        Path(MODEL, "text_encoder").glob("*.safetensors")
                    )
                ]
            ),
            Flux2ModelConfig(
                path=str(
                    Path(MODEL, "transformer", "diffusion_pytorch_model.safetensors")
                )
            ),
            Flux2ModelConfig(
                path=str(Path(MODEL, "vae", "diffusion_pytorch_model.safetensors"))
            ),
        ]
        tokenizer_config = Flux2ModelConfig(path=str(Path(MODEL, "tokenizer")))
    else:
        model_configs = [
            Flux2ModelConfig(
                model_id=MODEL, origin_file_pattern="text_encoder/*.safetensors"
            ),
            Flux2ModelConfig(
                model_id=MODEL, origin_file_pattern="transformer/*.safetensors"
            ),
            Flux2ModelConfig(
                model_id=MODEL,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ]
        tokenizer_config = Flux2ModelConfig(
            model_id=MODEL, origin_file_pattern="tokenizer/"
        )
    return Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )


def load_qwen_pipeline() -> Pipeline:
    return QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            QwenModelConfig(
                model_id=MODEL,
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            ),
            QwenModelConfig(
                model_id=QWEN_BASE_MODEL,
                origin_file_pattern="text_encoder/model*.safetensors",
            ),
            QwenModelConfig(
                model_id=QWEN_BASE_MODEL,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            ),
        ],
        processor_config=QwenModelConfig(
            model_id=QWEN_PROCESSOR_MODEL,
            origin_file_pattern="processor/",
        ),
    )


def load_pipeline() -> Pipeline:
    if PIPELINE_KIND == PIPELINE_KIND_FLUX2:
        return load_flux2_pipeline()
    return load_qwen_pipeline()


def generation_kwargs(item: WorkItem) -> JsonDict:
    key = item.key
    kwargs: JsonDict = {
        "seed": item.seed,
        "num_inference_steps": key.steps,
        "cfg_scale": key.cfg_scale if ENABLE_CFG else 1.0,
    }
    if key.height is not None:
        kwargs["height"] = key.height
    if key.width is not None:
        kwargs["width"] = key.width
    return kwargs


def run_qwen_pipeline_one(pipe: Pipeline, item: WorkItem) -> Image.Image:
    kwargs = generation_kwargs(item)
    kwargs["negative_prompt"] = item.key.negative_prompt
    kwargs["edit_image"] = [item.image]
    kwargs["edit_image_auto_resize"] = True
    kwargs["zero_cond_t"] = True
    return pipe(item.prompt, **kwargs)


def run_flux2_pipeline_one(pipe: Pipeline, item: WorkItem) -> Image.Image:
    kwargs = generation_kwargs(item)
    kwargs["negative_prompt"] = item.key.negative_prompt
    kwargs["edit_image"] = [item.image]
    kwargs["rand_device"] = "cuda"
    return pipe(item.prompt, **kwargs)


def run_pipeline_one(pipe: Pipeline, item: WorkItem) -> Image.Image:
    with torch.inference_mode():
        if PIPELINE_KIND == PIPELINE_KIND_FLUX2:
            return run_flux2_pipeline_one(pipe, item)
        return run_qwen_pipeline_one(pipe, item)


def process_item(pipe: Pipeline, item: WorkItem) -> None:
    started = time.perf_counter()
    try:
        image = run_pipeline_one(pipe, item)
        item.result = image_to_png_bytes(image)
        item.done.set()
        logging.info(
            "generated steps=%d enable_cfg=%s elapsed=%.3fs",
            item.key.steps,
            ENABLE_CFG,
            time.perf_counter() - started,
        )
    except Exception as exc:
        item.error = exc
        item.done.set()


def worker_main(pipe: Pipeline) -> None:
    while True:
        process_item(pipe, work_queue.get())


def start_worker(pipe: Pipeline) -> None:
    thread = threading.Thread(
        target=worker_main,
        args=(pipe,),
        name="diffsynth-image-worker",
        daemon=True,
    )
    thread.start()


def load_pipeline_or_exit() -> Pipeline:
    logging.info("loading %s pipeline from %s", PIPELINE_KIND, MODEL)
    pipe = load_pipeline()
    logging.info(
        "model loaded, pipeline=%s enable_cfg=%s default_steps=%d",
        PIPELINE_KIND,
        ENABLE_CFG,
        DEFAULT_STEPS,
    )
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--qwen-base-model", default=QWEN_BASE_MODEL)
    parser.add_argument("--qwen-processor-model", default=QWEN_PROCESSOR_MODEL)
    parser.add_argument("--enable-cfg", action="store_true")
    parser.add_argument("--steps", type=int, required=True)
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global MODEL, QWEN_BASE_MODEL, QWEN_PROCESSOR_MODEL
    global PIPELINE_KIND, ENABLE_CFG, DEFAULT_STEPS
    if not args.model:
        raise ValueError("--model must be non-empty")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    MODEL = args.model
    QWEN_BASE_MODEL = args.qwen_base_model
    QWEN_PROCESSOR_MODEL = args.qwen_processor_model
    PIPELINE_KIND = infer_pipeline_kind(args.model)
    ENABLE_CFG = args.enable_cfg
    DEFAULT_STEPS = args.steps


if __name__ == "__main__":
    configure_from_args(parse_args())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    pipeline = load_pipeline_or_exit()
    start_worker(pipeline)
    app.run(host=HOST, port=PORT, threaded=True)
