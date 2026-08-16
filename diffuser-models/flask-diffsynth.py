import argparse
import base64
import io
import logging
import os
import queue
import threading
import time
import typing as ty
from dataclasses import dataclass, field
from pathlib import Path

import torch
from diffsynth.pipelines.flux2_image import (
    Flux2ImagePipeline,
    ModelConfig as Flux2ModelConfig,
)
from diffsynth.pipelines.qwen_image import (
    ModelConfig as QwenModelConfig,
    QwenImagePipeline,
)
from flask import Flask, request
from PIL import Image

r"""
Server arguments:
| Name                   | Required | Description                                      |
| ---------------------- | -------- | ------------------------------------------------ |
| --model                | yes      | DiffSynth model directory or model ID.           |
| --enable-cfg           | no       | Enables CFG; requires --guidance-scale.          |
| --guidance-scale       | with CFG | Server-wide CFG scale when CFG is enabled.       |
| --steps                | yes      | Server-wide inference step count.                |

POST /v1/images/edits multipart/form-data fields:
| Name   | Required | Description                                  |
| ------ | -------- | -------------------------------------------- |
| image  | yes      | Source image file.                           |
| prompt | yes      | Edit instruction.                            |
| mask   | no       | Mask image file; must match the image size.  |
| n      | no       | Output image count; only n=1 is supported.   |

Responses always use b64_json.
"""

PIPELINE_KIND_QWEN = "qwen"
PIPELINE_KIND_FLUX2 = "flux2"
HOST = "0.0.0.0"
PORT = 8512
EDIT_IMAGE_AUTO_RESIZE = False


@dataclass(frozen=True)
class ServerConfig:
    model: str
    pipeline_kind: str
    enable_cfg: bool
    steps: int
    guidance_scale: float


@dataclass(frozen=True)
class WorkKey:
    steps: int
    cfg_scale: float
    height: int
    width: int


@dataclass
class WorkItem:
    image: Image.Image
    mask: ty.Optional[Image.Image]
    prompt: str
    key: WorkKey
    request_image_bytes: int
    done: threading.Event = field(default_factory=threading.Event)
    result: ty.Optional[bytes] = None
    error: ty.Optional[BaseException] = None


JsonDict = ty.Dict[str, ty.Any]
Pipeline = ty.Any


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


def infer_pipeline_kind(model: str) -> str:
    normalized = model.lower()
    if "flux2" in normalized or "flux.2" in normalized or "flux_2" in normalized:
        return PIPELINE_KIND_FLUX2
    return PIPELINE_KIND_QWEN


class DiffSynthEditServer:
    """Runs the Flask image-edit API, DiffSynth pipeline, and worker queue."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.app = Flask(__name__)
        self.work_queue: "queue.Queue[WorkItem]" = queue.Queue()
        self.app.post("/v1/images/edits")(self.edit_image)

    def build_item_from_values(
        self,
        image: Image.Image,
        mask: ty.Optional[Image.Image],
        request_image_bytes: int,
        values: ty.Mapping[str, ty.Any],
    ) -> WorkItem:
        if mask is not None and mask.size != image.size:
            raise ValueError(
                f"mask size {mask.size[0]}x{mask.size[1]} must match image size "
                f"{image.size[0]}x{image.size[1]}"
            )

        prompt = parse_str(values, "prompt", "")
        if not prompt:
            raise ValueError("missing prompt")

        return WorkItem(
            image=image,
            mask=mask,
            prompt=prompt,
            key=WorkKey(
                steps=self.config.steps,
                cfg_scale=self.config.guidance_scale,
                height=image.height,
                width=image.width,
            ),
            request_image_bytes=request_image_bytes,
        )

    def enqueue_and_wait(self, item: WorkItem) -> bytes:
        self.work_queue.put(item)
        item.done.wait()

        if item.error is not None:
            raise item.error
        if item.result is None:
            raise RuntimeError("empty generation result")
        return item.result

    def edit_image(self) -> ty.Any:
        try:
            image_file = request.files.get("image")
            if image_file is None:
                raise ValueError("missing image")
            image_bytes = image_file.stream.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            mask = None
            mask_file = request.files.get("mask")
            if mask_file is not None:
                mask = Image.open(mask_file.stream).convert("L")

            item = self.build_item_from_values(
                image,
                mask,
                len(image_bytes),
                request.form,
            )
            n = parse_int(request.form, "n", 1)
            if n != 1:
                raise ValueError("only n=1 is supported")
        except Exception as exc:
            return {
                "error": {"message": str(exc), "type": "invalid_request_error"}
            }, 400

        try:
            result = self.enqueue_and_wait(item)
        except Exception as exc:
            logging.error("image edit failed: %r", exc)
            return {"error": {"message": repr(exc), "type": "server_error"}}, 500

        b64_json = base64.b64encode(result).decode("ascii")
        image_object: JsonDict = {"b64_json": b64_json, "revised_prompt": item.prompt}
        return {"created": int(time.time()), "data": [image_object]}

    def load_flux2_pipeline(self) -> Pipeline:
        if os.path.isdir(self.config.model):
            model_configs = [
                Flux2ModelConfig(
                    path=[
                        str(path)
                        for path in sorted(
                            Path(self.config.model, "text_encoder").glob(
                                "*.safetensors"
                            )
                        )
                    ]
                ),
                Flux2ModelConfig(
                    path=str(
                        Path(
                            self.config.model,
                            "transformer",
                            "diffusion_pytorch_model.safetensors",
                        )
                    )
                ),
                Flux2ModelConfig(
                    path=str(
                        Path(
                            self.config.model,
                            "vae",
                            "diffusion_pytorch_model.safetensors",
                        )
                    )
                ),
            ]
            tokenizer_config = Flux2ModelConfig(
                path=str(Path(self.config.model, "tokenizer"))
            )
        else:
            model_configs = [
                Flux2ModelConfig(
                    model_id=self.config.model,
                    origin_file_pattern="text_encoder/*.safetensors",
                ),
                Flux2ModelConfig(
                    model_id=self.config.model,
                    origin_file_pattern="transformer/*.safetensors",
                ),
                Flux2ModelConfig(
                    model_id=self.config.model,
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                ),
            ]
            tokenizer_config = Flux2ModelConfig(
                model_id=self.config.model,
                origin_file_pattern="tokenizer/",
            )
        return Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

    def load_qwen_pipeline(self) -> Pipeline:
        if os.path.isdir(self.config.model):
            transformer_config = QwenModelConfig(
                path=[
                    str(path)
                    for path in sorted(
                        Path(self.config.model, "transformer").glob(
                            "diffusion_pytorch_model*.safetensors"
                        )
                    )
                ]
            )
        else:
            transformer_config = QwenModelConfig(
                model_id=self.config.model,
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
            )

        if os.path.isdir(self.config.model):
            text_encoder_config = QwenModelConfig(
                path=[
                    str(path)
                    for path in sorted(
                        Path(self.config.model, "text_encoder").glob(
                            "model*.safetensors"
                        )
                    )
                ]
            )
            vae_config = QwenModelConfig(
                path=str(
                    Path(
                        self.config.model,
                        "vae",
                        "diffusion_pytorch_model.safetensors",
                    )
                )
            )
        else:
            text_encoder_config = QwenModelConfig(
                model_id=self.config.model,
                origin_file_pattern="text_encoder/model*.safetensors",
            )
            vae_config = QwenModelConfig(
                model_id=self.config.model,
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            )

        if os.path.isdir(self.config.model):
            processor_config = QwenModelConfig(
                path=str(Path(self.config.model, "processor"))
            )
        else:
            processor_config = QwenModelConfig(
                model_id=self.config.model,
                origin_file_pattern="processor/",
            )

        return QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                transformer_config,
                text_encoder_config,
                vae_config,
            ],
            processor_config=processor_config,
        )

    def generation_kwargs(self, item: WorkItem) -> JsonDict:
        return {
            "num_inference_steps": item.key.steps,
            "cfg_scale": item.key.cfg_scale if self.config.enable_cfg else 1.0,
            "height": item.key.height,
            "width": item.key.width,
        }

    def run_qwen_pipeline_one(self, pipe: Pipeline, item: WorkItem) -> Image.Image:
        kwargs = self.generation_kwargs(item)
        kwargs["edit_image"] = [item.image]
        if item.mask is not None:
            kwargs["input_image"] = item.image
            kwargs["inpaint_mask"] = item.mask
        kwargs["edit_image_auto_resize"] = EDIT_IMAGE_AUTO_RESIZE
        kwargs["zero_cond_t"] = True
        return pipe(item.prompt, **kwargs)

    def run_flux2_pipeline_one(self, pipe: Pipeline, item: WorkItem) -> Image.Image:
        kwargs = self.generation_kwargs(item)
        kwargs["edit_image"] = [item.image]
        kwargs["edit_image_auto_resize"] = EDIT_IMAGE_AUTO_RESIZE
        if item.mask is not None:
            kwargs["input_image"] = item.image
            kwargs["inpaint_mask"] = item.mask
        kwargs["rand_device"] = "cuda"
        return pipe(item.prompt, **kwargs)

    def run_pipeline_one(self, pipe: Pipeline, item: WorkItem) -> Image.Image:
        with torch.inference_mode():
            if self.config.pipeline_kind == PIPELINE_KIND_FLUX2:
                return self.run_flux2_pipeline_one(pipe, item)
            return self.run_qwen_pipeline_one(pipe, item)

    def process_item(self, pipe: Pipeline, item: WorkItem) -> None:
        started = time.perf_counter()
        try:
            image = self.run_pipeline_one(pipe, item)
            item.result = image_to_png_bytes(image)
            item.done.set()
            logging.info(
                "generated steps=%d enable_cfg=%s mask=%s edit_image_auto_resize=%s size=%dx%d "
                "request_image_bytes=%d result_size=%dx%d elapsed=%.3fs",
                item.key.steps,
                self.config.enable_cfg,
                item.mask is not None,
                EDIT_IMAGE_AUTO_RESIZE,
                item.key.width,
                item.key.height,
                item.request_image_bytes,
                image.width,
                image.height,
                time.perf_counter() - started,
            )
        except Exception as exc:
            item.error = exc
            item.done.set()

    def worker_main(self, pipe: Pipeline) -> None:
        while True:
            self.process_item(pipe, self.work_queue.get())

    def start_worker(self, pipe: Pipeline) -> None:
        thread = threading.Thread(
            target=self.worker_main,
            args=(pipe,),
            name="diffsynth-image-worker",
            daemon=True,
        )
        thread.start()

    def load_pipeline_or_exit(self) -> Pipeline:
        logging.info(
            "loading %s pipeline from %s",
            self.config.pipeline_kind,
            self.config.model,
        )
        if self.config.pipeline_kind == PIPELINE_KIND_FLUX2:
            pipe = self.load_flux2_pipeline()
        else:
            pipe = self.load_qwen_pipeline()
        logging.info("compiling pipeline models")
        pipe.compile_pipeline(
            mode="default",
            dynamic=True,
            fullgraph=False,
        )
        logging.info(
            "model loaded, pipeline=%s enable_cfg=%s default_steps=%d",
            self.config.pipeline_kind,
            self.config.enable_cfg,
            self.config.steps,
        )
        return pipe

    def run(self) -> None:
        pipe = self.load_pipeline_or_exit()
        self.start_worker(pipe)
        self.app.run(host=HOST, port=PORT, threaded=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--enable-cfg", action="store_true")
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--steps", type=int, required=True)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ServerConfig:
    if not args.model:
        raise ValueError("--model must be non-empty")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.enable_cfg and args.guidance_scale is None:
        raise ValueError("--guidance-scale is required when --enable-cfg is set")
    return ServerConfig(
        model=args.model,
        pipeline_kind=infer_pipeline_kind(args.model),
        enable_cfg=args.enable_cfg,
        steps=args.steps,
        guidance_scale=args.guidance_scale if args.guidance_scale is not None else 1.0,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    server = DiffSynthEditServer(config_from_args(parse_args()))
    server.run()
