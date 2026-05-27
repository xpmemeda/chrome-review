from __future__ import annotations

import argparse
import gc
import math
import random
import statistics
import time

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import av
from av.codec.hwaccel import HWAccel
from decode_heif import HeifDecoder
from PIL import Image
from pillow_heif import open_heif

DEFAULT_TTHEIF_LIB = "/home/tiger/local/libttheif/lib/shared/libttheif_dec.so"


class ImageDecoder(ABC):
    """Decodes one image through a named backend."""

    def __init__(self, name: str, source: Path) -> None:
        self.name = name
        self.source = source

    @property
    def hwaccel(self) -> bool | None:
        """Returns hardware acceleration status when the backend can report it."""
        return None

    @abstractmethod
    def decode(self) -> Image.Image:
        """Returns the decoded primary image as an RGB Pillow image."""


class PillowDecoder(ImageDecoder):
    """Decodes images using Pillow or pillow-heif based on the source suffix."""

    def __init__(self, source: Path) -> None:
        super().__init__("pillow", source)
        self._loader: Callable[[], Image.Image]
        if self._is_heif_source():
            self._loader = self._load_with_heif
        else:
            self._loader = self._load_with_pillow

    def decode(self) -> Image.Image:
        """Loads the source with the selected backend and normalizes it to RGB."""
        return self._loader()

    def _load_with_pillow(self) -> Image.Image:
        """Loads a Pillow-supported image and normalizes it to RGB."""
        with Image.open(self.source) as image:
            image.load()
            return image.convert("RGB")

    def _load_with_heif(self) -> Image.Image:
        """Loads a HEIF image through pillow-heif and normalizes it to RGB."""
        heif = open_heif(str(self.source), thumbnails=False)
        return Image.frombytes(
            heif.mode,
            heif.size,
            heif.data,
            "raw",
        ).convert("RGB")

    def _is_heif_source(self) -> bool:
        """Returns whether the source filename looks like a HEIF container."""
        return self.source.suffix.lower() in {
            ".heic",
            ".heif",
            ".avif",
            ".hif",
        }


class TtheifDecoder(ImageDecoder):
    """Decodes HEIC images using libttheif."""

    def __init__(self, source: Path, lib_path: str) -> None:
        super().__init__("ttheif", source)
        self.decoder = HeifDecoder(lib_path)

    def decode(self) -> Image.Image:
        """Runs libttheif and normalizes the decoded image to RGB."""
        image = self.decoder.decode(str(self.source))
        if image is None:
            raise RuntimeError(f"libttheif failed to decode {self.source}")
        return image.convert("RGB")


class PyavDecoder(ImageDecoder):
    """Decodes HEIC primary streams or grid tiles using PyAV."""

    def __init__(self, source: Path, name: str, accelerator: HWAccel | None) -> None:
        super().__init__(name, source)
        self.accelerator = accelerator
        self._image_size: tuple[int, int] | None = None
        self._hwaccel: bool | None = None

    @property
    def hwaccel(self) -> bool | None:
        """Returns whether streams used for output are hardware accelerated."""
        if self.accelerator is None:
            return None
        if self._hwaccel is None:
            self._hwaccel = self._is_hardware_decode_enabled()
        return self._hwaccel

    def decode(self) -> Image.Image:
        """Decodes a HEIC primary image or its grid tiles through PyAV."""
        image_size = self._get_heif_size()
        open_options = (
            {"hwaccel": self.accelerator} if self.accelerator is not None else {}
        )
        with av.open(str(self.source), **open_options) as container:
            streams = list(container.streams.video)
            dependent = self._dependent_disposition(streams)
            primary_decode_error = None
            for stream in streams:
                if self._stream_size(stream) != image_size:
                    continue
                if self._is_dependent_stream(stream, dependent):
                    continue
                try:
                    return self._decode_stream(container, stream)
                except Exception as exc:
                    primary_decode_error = exc

            tiles = [
                stream
                for stream in streams
                if self._is_dependent_stream(stream, dependent)
            ]
            if not tiles:
                if primary_decode_error is not None:
                    raise RuntimeError(
                        f"PyAV failed to decode the primary image stream for {image_size}"
                    ) from primary_decode_error
                raise RuntimeError(
                    f"PyAV found no image stream or grid tiles for {image_size}"
                )

            tile_size = self._stream_size(tiles[0])
            if tile_size is None:
                raise RuntimeError("PyAV exposed grid tiles without a valid tile size")
            tile_width, tile_height = tile_size
            columns = math.ceil(image_size[0] / tile_width)
            rows = math.ceil(image_size[1] / tile_height)
            required_tiles = columns * rows
            if len(tiles) < required_tiles:
                raise RuntimeError(
                    f"HEIF grid needs {required_tiles} tiles for {image_size}, "
                    f"but PyAV exposed {len(tiles)}"
                )

            image = Image.new("RGB", (columns * tile_width, rows * tile_height))
            for position, stream in enumerate(tiles[:required_tiles]):
                tile = self._decode_stream(container, stream)
                x = (position % columns) * tile_width
                y = (position // columns) * tile_height
                image.paste(tile, (x, y))
            return image.crop((0, 0, image_size[0], image_size[1]))

    def _get_heif_size(self) -> tuple[int, int]:
        """Returns the final HEIF presentation size."""
        if self._image_size is None:
            self._image_size = open_heif(str(self.source)).size
        return self._image_size

    def _is_hardware_decode_enabled(self) -> bool:
        """Reports whether all PyAV streams used for the output use hwaccel."""
        image_size = self._get_heif_size()
        with av.open(str(self.source), hwaccel=self.accelerator) as container:
            streams = list(container.streams.video)
            dependent = self._dependent_disposition(streams)
            matching = [
                stream
                for stream in streams
                if self._stream_size(stream) == image_size
                and not self._is_dependent_stream(stream, dependent)
            ]
            if matching:
                return all(stream.codec_context.is_hwaccel for stream in matching)

            tiles = [
                stream
                for stream in streams
                if self._is_dependent_stream(stream, dependent)
            ]
            return bool(tiles) and all(
                stream.codec_context.is_hwaccel for stream in tiles
            )

    def _decode_stream(self, container: Any, stream: Any) -> Image.Image:
        """Decodes one PyAV stream and copies its first frame to a host RGB image."""
        for frame in container.decode(stream):
            return frame.to_image().convert("RGB")
        raise RuntimeError(
            f"PyAV did not return an image frame for stream {stream.index}"
        )

    @staticmethod
    def _stream_size(stream: Any) -> tuple[int, int] | None:
        """Returns a PyAV stream's coded size when it is available."""
        context = stream.codec_context
        if context.width <= 0 or context.height <= 0:
            return None
        return (context.width, context.height)

    @staticmethod
    def _dependent_disposition(streams: list[Any]) -> int | None:
        """Returns PyAV's dependent stream disposition flag."""
        return type(streams[0].disposition).dependent if streams else None

    @staticmethod
    def _is_dependent_stream(stream: Any, dependent: int | None) -> bool:
        """Returns whether a PyAV stream is marked as a dependent item."""
        return dependent is not None and bool(stream.disposition & dependent)


@dataclass(frozen=True)
class Measurements:
    """Latency samples and validated output size for a benchmark run."""

    samples_ms: dict[str, list[float]]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class Result:
    """Latency measurements and decode-path metadata for one provider."""

    provider: str
    samples_ms: list[float]
    image_size: tuple[int, int]
    hwaccel: bool | None

    def to_dictionary(self) -> dict[str, object]:
        """Returns a JSON-serializable benchmark result."""
        ordered = sorted(self.samples_ms)
        index_95 = math.ceil(len(ordered) * 0.95) - 1
        return {
            "provider": self.provider,
            "runs": len(self.samples_ms),
            "image_size": list(self.image_size),
            "hwaccel": self.hwaccel,
            "mean_ms": statistics.mean(self.samples_ms),
            "median_ms": statistics.median(self.samples_ms),
            "min_ms": min(self.samples_ms),
            "p95_ms": ordered[index_95],
            "max_ms": max(self.samples_ms),
            "samples_ms": self.samples_ms,
        }


def validate_image_size(
    name: str,
    image: Image.Image,
    image_size: tuple[int, int] | None,
) -> tuple[int, int]:
    """Validates that all providers return the same output image size."""
    if image_size is None:
        return image.size
    if image.size != image_size:
        raise RuntimeError(
            f"{name} returned image size {image.size}, expected {image_size}"
        )
    return image_size


def measure(
    decoders: list[ImageDecoder],
    warmup: int,
    runs: int,
    seed: int,
) -> Measurements:
    """Measures providers in shuffled round-robin order after warmup iterations."""
    decoders = list(decoders)
    rng = random.Random(seed)
    image_size = None
    for _ in range(warmup):
        rng.shuffle(decoders)
        for decoder in decoders:
            image_size = validate_image_size(decoder.name, decoder.decode(), image_size)

    samples = {decoder.name: [] for decoder in decoders}
    gc.disable()
    try:
        for _ in range(runs):
            rng.shuffle(decoders)
            for decoder in decoders:
                start_ns = time.perf_counter_ns()
                image = decoder.decode()
                elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                samples[decoder.name].append(elapsed_ms)
                image_size = validate_image_size(decoder.name, image, image_size)
    finally:
        gc.enable()
    if image_size is None:
        raise RuntimeError("benchmark did not decode any image")
    return Measurements(samples, image_size)


def print_results(results: list[Result]) -> None:
    """Prints compact benchmark statistics in milliseconds."""
    print("provider   hwaccel  runs  mean_ms  median_ms  min_ms  p95_ms  max_ms")
    for result in results:
        summary = result.to_dictionary()
        hwaccel = "-" if result.hwaccel is None else str(result.hwaccel).lower()
        print(
            f"{result.provider:<10} {hwaccel:<8} {summary['runs']:>4} "
            f"{summary['mean_ms']:>8.3f} {summary['median_ms']:>10.3f} "
            f"{summary['min_ms']:>7.3f} {summary['p95_ms']:>7.3f} {summary['max_ms']:>7.3f}"
        )


def main() -> None:
    """Parses CLI options and executes the decode benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark HEIC decode to an in-memory RGB Pillow image."
    )
    parser.add_argument("--image", type=Path, help="input image")
    parser.add_argument(
        "--tool",
        type=str,
        help="The decoder to use, e.g. pillow, pyav-sw, pyav-hw, or ttheif",
    )
    parser.add_argument(
        "--ttheif-lib",
        default=DEFAULT_TTHEIF_LIB,
        help=f"libttheif decoder library path (default: {DEFAULT_TTHEIF_LIB})",
    )
    parser.add_argument(
        "--device", default="cuda", help="PyAV HWAccel device type (default: cuda)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="warmup runs per provider (default: 3)"
    )
    parser.add_argument(
        "--runs", type=int, default=20, help="measured runs per provider (default: 20)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="provider ordering seed (default: 0)"
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.runs <= 0:
        parser.error("--warmup must be non-negative and --runs must be positive")

    if args.tool == "pillow":
        decoders = [PillowDecoder(args.image)]
    elif args.tool == "ttheif":
        decoders = [TtheifDecoder(args.image, args.ttheif_lib)]
    elif args.tool == "pyav-sw":
        decoders = [PyavDecoder(args.image, "pyav-sw", None)]
    elif args.tool == "pyav-hw":
        accelerator = HWAccel(device_type=args.device, allow_software_fallback=False)
        decoders = [PyavDecoder(args.image, "pyav-hw", accelerator)]
    else:
        parser.error(f"Unsupported tool: {args.tool}")

    measurements = measure(decoders, args.warmup, args.runs, args.seed)
    results = [
        Result(
            decoder.name,
            measurements.samples_ms[decoder.name],
            measurements.image_size,
            decoder.hwaccel,
        )
        for decoder in decoders
    ]
    print_results(results)
    if args.tool == "pyav-hw" and not decoders[0].hwaccel:
        print(
            "WARNING: pyav-hw did not report hardware-accelerated decoding for the selected image streams."
        )


if __name__ == "__main__":
    main()
