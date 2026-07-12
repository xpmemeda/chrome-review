import argparse
import io
import logging
import queue
import threading
import time
import typing as ty
from dataclasses import dataclass, field

import torch
from diffusers import QwenImageEditPlusPipeline
from flask import Flask, request, send_file
from PIL import Image


MODEL = "/home/xiongpeng/workspace/models/Qwen-Image-Edit-2511"
HOST = "0.0.0.0"
PORT = 8512

MAX_BATCH_SIZE = 0
MAX_WAIT_MS = 0.0
DEFAULT_STEPS = 0
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_GUIDANCE_SCALE = 1.0


app = Flask(__name__)
work_queue: "queue.Queue[BatchItem]" = queue.Queue()


@dataclass(frozen=True)
class BatchKey:
    steps: int
    true_cfg_scale: float
    guidance_scale: float
    negative_prompt: str
    size: ty.Tuple[int, int]


@dataclass
class BatchItem:
    image: Image.Image
    prompt: str
    seed: int
    key: BatchKey
    done: threading.Event = field(default_factory=threading.Event)
    result: ty.Optional[bytes] = None
    error: ty.Optional[BaseException] = None


def parse_int(name: str, default: int) -> int:
    value = request.form.get(name)
    return int(value) if value not in (None, "") else default


def parse_float(name: str, default: float) -> float:
    value = request.form.get(name)
    return float(value) if value not in (None, "") else default


def build_item() -> BatchItem:
    if "image" not in request.files:
        raise ValueError("missing image")

    image = Image.open(request.files["image"].stream).convert("RGB")
    style = request.form.get("style") or "摄影后期"
    prompt = (
        request.form.get("prompt") or f"请对图片进行{style}风格的编辑，保持主体一致。"
    )
    seed = parse_int("seed", 0)
    steps = parse_int("steps", DEFAULT_STEPS)
    true_cfg_scale = parse_float("true_cfg_scale", DEFAULT_TRUE_CFG_SCALE)
    guidance_scale = parse_float("guidance_scale", DEFAULT_GUIDANCE_SCALE)
    negative_prompt = request.form.get("negative_prompt") or " "

    return BatchItem(
        image=image,
        prompt=prompt,
        seed=seed,
        key=BatchKey(
            steps=steps,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            size=image.size,
        ),
    )


@app.post("/generate")
def generate():
    try:
        item = build_item()
    except Exception as exc:
        return {"error": str(exc)}, 400

    work_queue.put(item)
    item.done.wait()

    if item.error is not None:
        logging.error("generation failed: %r", item.error)
        return {"error": repr(item.error)}, 500
    if item.result is None:
        return {"error": "empty generation result"}, 500

    return send_file(io.BytesIO(item.result), mimetype="image/png")


def collect_batch(first: BatchItem, pending: ty.List[BatchItem]) -> ty.List[BatchItem]:
    batch = [first]
    deadline = time.perf_counter() + MAX_WAIT_MS / 1000.0

    i = 0
    while i < len(pending) and len(batch) < MAX_BATCH_SIZE:
        item = pending[i]
        if item.key == first.key:
            batch.append(item)
            pending.pop(i)
        else:
            i += 1

    while len(batch) < MAX_BATCH_SIZE:
        timeout = deadline - time.perf_counter()
        if timeout <= 0:
            break
        try:
            item = work_queue.get(timeout=timeout)
        except queue.Empty:
            break
        if item.key == first.key:
            batch.append(item)
        else:
            pending.append(item)

    return batch


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def make_generators(batch: ty.List[BatchItem]) -> ty.List[torch.Generator]:
    return [torch.Generator(device="cuda").manual_seed(item.seed) for item in batch]


def run_pipeline_batch(
    pipe: QwenImageEditPlusPipeline, batch: ty.List[BatchItem]
) -> ty.List[Image.Image]:
    key = batch[0].key
    with torch.inference_mode():
        out = pipe(
            image=[item.image for item in batch],
            prompt=[item.prompt for item in batch],
            generator=make_generators(batch),
            true_cfg_scale=key.true_cfg_scale,
            negative_prompt=[key.negative_prompt] * len(batch),
            num_inference_steps=key.steps,
            guidance_scale=key.guidance_scale,
            num_images_per_prompt=1,
        )
    images = list(out.images)
    if len(images) != len(batch):
        raise RuntimeError(
            f"pipeline returned {len(images)} images for batch size {len(batch)}"
        )
    return images


def run_pipeline_one(pipe: QwenImageEditPlusPipeline, item: BatchItem) -> Image.Image:
    key = item.key
    with torch.inference_mode():
        out = pipe(
            image=[item.image],
            prompt=item.prompt,
            generator=torch.Generator(device="cuda").manual_seed(item.seed),
            true_cfg_scale=key.true_cfg_scale,
            negative_prompt=key.negative_prompt,
            num_inference_steps=key.steps,
            guidance_scale=key.guidance_scale,
            num_images_per_prompt=1,
        )
    return out.images[0]


def finish_success(batch: ty.List[BatchItem], images: ty.List[Image.Image]) -> None:
    for item, image in zip(batch, images):
        item.result = image_to_png_bytes(image)
        item.done.set()


def finish_error(batch: ty.List[BatchItem], exc: BaseException) -> None:
    for item in batch:
        item.error = exc
        item.done.set()


def process_batch(pipe: QwenImageEditPlusPipeline, batch: ty.List[BatchItem]) -> None:
    started = time.perf_counter()
    try:
        if len(batch) == 1:
            images = [run_pipeline_one(pipe, batch[0])]
        else:
            try:
                images = run_pipeline_batch(pipe, batch)
            except Exception:
                logging.exception(
                    "batched generation failed; falling back to sequential generation"
                )
                images = [run_pipeline_one(pipe, item) for item in batch]
        finish_success(batch, images)
        logging.info(
            "generated batch size=%d steps=%d elapsed=%.3fs",
            len(batch),
            batch[0].key.steps,
            time.perf_counter() - started,
        )
    except Exception as exc:
        finish_error(batch, exc)


def worker_main() -> None:
    logging.info("loading model from %s", MODEL)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    logging.info(
        "model loaded, max_batch_size=%d max_wait_ms=%.1f default_steps=%d",
        MAX_BATCH_SIZE,
        MAX_WAIT_MS,
        DEFAULT_STEPS,
    )

    pending: ty.List[BatchItem] = []
    while True:
        if pending:
            first = pending.pop(0)
        else:
            first = work_queue.get()
        process_batch(pipe, collect_batch(first, pending))


def start_worker() -> None:
    thread = threading.Thread(target=worker_main, name="qwen-image-worker", daemon=True)
    thread.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batch-size", type=int, required=True)
    parser.add_argument("--max-wait-ms", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global MAX_BATCH_SIZE, MAX_WAIT_MS, DEFAULT_STEPS
    if args.max_batch_size <= 0:
        raise ValueError("--max-batch-size must be positive")
    if args.max_wait_ms < 0:
        raise ValueError("--max-wait-ms must be non-negative")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    MAX_BATCH_SIZE = args.max_batch_size
    MAX_WAIT_MS = args.max_wait_ms
    DEFAULT_STEPS = args.steps


if __name__ == "__main__":
    configure_from_args(parse_args())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    start_worker()
    app.run(host=HOST, port=PORT, threaded=True)
