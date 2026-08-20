import argparse
import base64
import collections
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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    fp8_dit: bool
    fp8_text_encoder: bool
    profile_first_request: bool
    text_encoder_model: ty.Optional[str]
    compile_mode: str
    compile_vae: bool
    prompt_cache_size: int
    png_compress_level: int
    overlap_png_encode: bool
    log_stage_timing: bool


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
    result_image: ty.Optional[Image.Image] = None
    error: ty.Optional[BaseException] = None


JsonDict = ty.Dict[str, ty.Any]
Pipeline = ty.Any


def parse_int(values: ty.Mapping[str, ty.Any], name: str, default: int) -> int:
    value = values.get(name)
    return int(value) if value not in (None, "") else default


def parse_str(values: ty.Mapping[str, ty.Any], name: str, default: str) -> str:
    value = values.get(name)
    return str(value) if value not in (None, "") else default


def image_to_png_bytes(image: Image.Image, compress_level: int = 6) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=compress_level)
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
        self.profile_pending = config.profile_first_request
        self.prompt_cache: "collections.OrderedDict[ty.Hashable, JsonDict]" = (
            collections.OrderedDict()
        )
        self.prompt_cache_hits = 0
        self.prompt_cache_misses = 0
        self.app.post("/v1/images/edits")(self.edit_image)

    @staticmethod
    def quantize_fp8_w8a8(module: torch.nn.Module, name: str) -> None:
        """Quantize Linear weights and activations to E4M3 FP8 using real CUDA kernels."""
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            quantize_,
        )

        logging.info("quantizing %s with TorchAO dynamic FP8 W8A8", name)
        quantize_(module, Float8DynamicActivationFloat8WeightConfig())

        quantized = []
        for module_name, child in module.named_modules():
            if isinstance(child, torch.nn.Linear):
                weight_type = type(child.weight).__name__
                if "Float8" in weight_type:
                    quantized.append((module_name, weight_type, child.weight.dtype))
        if not quantized:
            raise RuntimeError(
                f"FP8 quantization produced no FP8 Linear weights in {name}"
            )
        logging.info(
            "FP8 audit component=%s quantized_linears=%d sample=%s",
            name,
            len(quantized),
            quantized[:3],
        )

    @staticmethod
    def audit_native_fp8(module: torch.nn.Module, name: str) -> None:
        fp8_modules = [
            (module_name, type(child).__name__)
            for module_name, child in module.named_modules()
            if "fp8" in type(child).__name__.lower()
        ]
        fp8_tensors = [
            (tensor_name, str(tensor.dtype), tuple(tensor.shape))
            for tensor_name, tensor in module.state_dict().items()
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        ]
        scale_tensors = [
            (tensor_name, str(tensor.dtype), tuple(tensor.shape))
            for tensor_name, tensor in module.state_dict().items()
            if "scale" in tensor_name.lower()
        ]
        logging.info(
            "native FP8 audit component=%s fp8_modules=%d fp8_tensors=%d scale_tensors=%d "
            "module_sample=%s tensor_sample=%s scale_sample=%s",
            name,
            len(fp8_modules),
            len(fp8_tensors),
            len(scale_tensors),
            fp8_modules[:3],
            fp8_tensors[:3],
            scale_tensors[:3],
        )

    @staticmethod
    def log_cuda_memory(stage: str) -> None:
        torch.cuda.synchronize()
        logging.info(
            "CUDA memory stage=%s allocated=%.3fGiB reserved=%.3fGiB peak_allocated=%.3fGiB",
            stage,
            torch.cuda.memory_allocated() / 2**30,
            torch.cuda.memory_reserved() / 2**30,
            torch.cuda.max_memory_allocated() / 2**30,
        )

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
        if self.config.overlap_png_encode:
            if item.result_image is None:
                raise RuntimeError("empty generation result")
            encode_started = time.perf_counter()
            result = image_to_png_bytes(
                item.result_image,
                compress_level=self.config.png_compress_level,
            )
            if self.config.log_stage_timing:
                logging.info(
                    "overlapped png_encode=%.3fs", time.perf_counter() - encode_started
                )
            return result
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
                *(
                    [
                        Flux2ModelConfig(
                            path=[
                                str(path)
                                for path in sorted(
                                    Path(self.config.model, "text_encoder").glob(
                                        "*.safetensors"
                                    )
                                )
                            ]
                        )
                    ]
                    if self.config.text_encoder_model is None
                    else []
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
        else:
            model_configs = [
                *(
                    [
                        Flux2ModelConfig(
                            model_id=self.config.model,
                            origin_file_pattern="text_encoder/*.safetensors",
                        )
                    ]
                    if self.config.text_encoder_model is None
                    else []
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
        tokenizer_config = None
        if self.config.text_encoder_model is None:
            if os.path.isdir(self.config.model):
                tokenizer_config = Flux2ModelConfig(
                    path=str(Path(self.config.model, "tokenizer"))
                )
            else:
                tokenizer_config = Flux2ModelConfig(
                    model_id=self.config.model,
                    origin_file_pattern="tokenizer/",
                )

        pipe = Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        if self.config.text_encoder_model is not None:
            logging.info(
                "loading independent Flux2 text encoder model=%s",
                self.config.text_encoder_model,
            )
            pipe.text_encoder_qwen3 = AutoModelForCausalLM.from_pretrained(
                self.config.text_encoder_model,
                torch_dtype=None,
                device_map="cuda",
                local_files_only=os.path.isdir(self.config.text_encoder_model),
            )
            pipe.tokenizer = AutoTokenizer.from_pretrained(
                self.config.text_encoder_model,
                local_files_only=os.path.isdir(self.config.text_encoder_model),
            )
            self.audit_native_fp8(pipe.text_encoder_qwen3, "text_encoder")
        else:
            logging.info("using text encoder and tokenizer bundled in --model")
        if self.config.fp8_dit:
            self.quantize_fp8_w8a8(pipe.dit, "dit")
        if self.config.fp8_text_encoder:
            self.quantize_fp8_w8a8(pipe.text_encoder_qwen3, "text_encoder")
        return pipe

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

    def install_prompt_cache(self, pipe: Pipeline) -> None:
        if self.config.prompt_cache_size <= 0:
            return
        if self.config.pipeline_kind != PIPELINE_KIND_FLUX2:
            raise ValueError("--prompt-cache-size currently supports only FLUX.2")

        units = [
            unit
            for unit in pipe.units
            if unit.__class__.__name__ == "Flux2Unit_Qwen3PromptEmbedder"
        ]
        if len(units) != 1:
            raise RuntimeError(
                f"expected one Qwen3 prompt embedder unit, found {len(units)}"
            )
        unit = units[0]
        original_process = unit.process

        def cached_process(
            pipe_arg: Pipeline, prompt: ty.Union[str, ty.List[str]]
        ) -> JsonDict:
            key: ty.Hashable = tuple(prompt) if isinstance(prompt, list) else prompt
            cached = self.prompt_cache.get(key)
            if cached is not None:
                self.prompt_cache.move_to_end(key)
                self.prompt_cache_hits += 1
                return cached

            self.prompt_cache_misses += 1
            result = original_process(pipe_arg, prompt)
            self.prompt_cache[key] = result
            self.prompt_cache.move_to_end(key)
            while len(self.prompt_cache) > self.config.prompt_cache_size:
                self.prompt_cache.popitem(last=False)
            return result

        unit.process = cached_process
        logging.info(
            "enabled exact prompt embedding cache entries=%d",
            self.config.prompt_cache_size,
        )

    def process_item(self, pipe: Pipeline, item: WorkItem) -> None:
        started = time.perf_counter()
        try:
            if self.profile_pending:
                self.profile_pending = False
                logging.info("profiling first request for FP8 CUDA kernels")
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                ) as prof:
                    image = self.run_pipeline_one(pipe, item)
                fp8_events = [
                    event
                    for event in prof.key_averages()
                    if any(
                        token in event.key.lower() for token in ("scaled_mm", "float8")
                    )
                ]
                logging.info(
                    "FP8 profiler events=%s",
                    [
                        {
                            "key": event.key,
                            "calls": event.count,
                            "cuda_ms": event.device_time_total / 1000.0,
                        }
                        for event in fp8_events
                    ],
                )
                if (
                    self.config.fp8_dit or self.config.fp8_text_encoder
                ) and not fp8_events:
                    logging.warning(
                        "FP8 enabled but profiler found no scaled_mm/float8 events"
                    )
                top_cuda_events = sorted(
                    prof.key_averages(),
                    key=lambda event: event.device_time_total,
                    reverse=True,
                )[:20]
                logging.info(
                    "CUDA profiler top events=%s",
                    [
                        {
                            "key": event.key,
                            "calls": event.count,
                            "cuda_ms": event.device_time_total / 1000.0,
                        }
                        for event in top_cuda_events
                    ],
                )
            else:
                image = self.run_pipeline_one(pipe, item)
            pipeline_finished = time.perf_counter()
            if self.config.overlap_png_encode:
                # Wake the Flask request thread so CPU PNG encoding can overlap
                # the next queued GPU inference.
                item.result_image = image
                item.done.set()
            else:
                item.result = image_to_png_bytes(
                    image,
                    compress_level=self.config.png_compress_level,
                )
            png_finished = time.perf_counter()
            if not self.config.overlap_png_encode:
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
            if self.config.log_stage_timing:
                logging.info(
                    "stage timing pipeline=%.3fs png_encode=%.3fs total=%.3fs "
                    "prompt_cache_hits=%d prompt_cache_misses=%d",
                    pipeline_finished - started,
                    png_finished - pipeline_finished,
                    png_finished - started,
                    self.prompt_cache_hits,
                    self.prompt_cache_misses,
                )
            self.log_cuda_memory("after_request")
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
        self.install_prompt_cache(pipe)
        logging.info("compiling pipeline models")
        pipe.compile_pipeline(
            mode=self.config.compile_mode,
            dynamic=True,
            fullgraph=False,
        )
        if self.config.compile_vae:
            logging.info(
                "compiling VAE encode and decode mode=%s", self.config.compile_mode
            )
            pipe.vae.encode = torch.compile(
                pipe.vae.encode,
                mode=self.config.compile_mode,
                dynamic=True,
                fullgraph=False,
            )
            pipe.vae.decode = torch.compile(
                pipe.vae.decode,
                mode=self.config.compile_mode,
                dynamic=True,
                fullgraph=False,
            )
        # Online quantization briefly materializes BF16 and FP8 weights together.
        # Release those cached BF16 blocks before reporting steady-state memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logging.info(
            "model loaded, pipeline=%s enable_cfg=%s default_steps=%d",
            self.config.pipeline_kind,
            self.config.enable_cfg,
            self.config.steps,
        )
        self.log_cuda_memory("after_load_and_compile")
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
    parser.add_argument(
        "--text-encoder-model",
        help=(
            "Optional local path or model ID for an independent FLUX.2 Qwen3 "
            "encoder; when omitted, use text_encoder/ and tokenizer/ from --model."
        ),
    )
    parser.add_argument(
        "--fp8-dit",
        action="store_true",
        help="Use dynamic E4M3 FP8 activations and FP8 weights for the FLUX.2 DiT.",
    )
    parser.add_argument(
        "--fp8-text-encoder",
        action="store_true",
        help="Use dynamic E4M3 FP8 activations and FP8 weights for the Qwen3 text encoder.",
    )
    parser.add_argument(
        "--profile-first-request",
        action="store_true",
        help="Profile the first request and log scaled_mm/float8 CUDA operators.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
        help="torch.compile mode for the DiT and optionally VAE.",
    )
    parser.add_argument(
        "--compile-vae",
        action="store_true",
        help="Compile FLUX.2 VAE encode and decode in addition to the DiT.",
    )
    parser.add_argument(
        "--prompt-cache-size",
        type=int,
        default=0,
        help="Number of exact FLUX.2 prompt embeddings to retain on GPU; 0 disables caching.",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        choices=range(10),
        default=6,
        metavar="0..9",
        help="PNG compression level; Pillow's default is 6.",
    )
    parser.add_argument(
        "--log-stage-timing",
        action="store_true",
        help="Log pipeline and PNG encoding wall times for every request.",
    )
    parser.add_argument(
        "--overlap-png-encode",
        action="store_true",
        help="Encode PNG in Flask request threads so it can overlap the next GPU request.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ServerConfig:
    if not args.model:
        raise ValueError("--model must be non-empty")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.enable_cfg and args.guidance_scale is None:
        raise ValueError("--guidance-scale is required when --enable-cfg is set")
    if args.prompt_cache_size < 0:
        raise ValueError("--prompt-cache-size must be non-negative")
    pipeline_kind = infer_pipeline_kind(args.model)
    if (args.fp8_dit or args.fp8_text_encoder) and pipeline_kind != PIPELINE_KIND_FLUX2:
        raise ValueError("FP8 flags are currently supported only for FLUX.2")
    return ServerConfig(
        model=args.model,
        pipeline_kind=pipeline_kind,
        enable_cfg=args.enable_cfg,
        steps=args.steps,
        guidance_scale=args.guidance_scale if args.guidance_scale is not None else 1.0,
        fp8_dit=args.fp8_dit,
        fp8_text_encoder=args.fp8_text_encoder,
        profile_first_request=args.profile_first_request,
        text_encoder_model=(
            os.path.expanduser(args.text_encoder_model)
            if args.text_encoder_model
            else None
        ),
        compile_mode=args.compile_mode,
        compile_vae=args.compile_vae,
        prompt_cache_size=args.prompt_cache_size,
        png_compress_level=args.png_compress_level,
        overlap_png_encode=args.overlap_png_encode,
        log_stage_timing=args.log_stage_timing,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    server = DiffSynthEditServer(config_from_args(parse_args()))
    server.run()
