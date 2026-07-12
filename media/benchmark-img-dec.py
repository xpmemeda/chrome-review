from __future__ import annotations

import argparse
import gc
import math
import statistics
import time

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import av
from av.codec.hwaccel import HWAccel
from decode_heif import HeifDecoder
from PIL import Image, ImageChops, ImageStat
from pillow_heif import open_heif

DEFAULT_TTHEIF_LIB = "/home/tiger/local/libttheif/lib/shared/libttheif_dec.so"
MAX_PIXEL_DIFFERENCE = 3
MAX_MEAN_PIXEL_DIFFERENCE = 0.5
GROUNDTRUTH_SUFFIX = "pillow-groundtruth"


class ImageMetadataReader:
    """Reads image type and dimensions without decoding pixels."""

    _FTYP_HEADER_SIZE = 4096

    def type(self, source: str | bytes) -> str:
        """Returns the image type."""
        return self._type_from_header(self._read_header(source)) or "UNKNOWN"

    def size(self, source: str | bytes) -> tuple[int, int]:
        """Returns image dimensions without decoding pixels."""
        if self.is_heif_family(source):
            return open_heif(self._openable_source(source), thumbnails=False).size
        with Image.open(self._openable_source(source)) as image:
            return image.size

    def is_heic(self, source: str | bytes) -> bool:
        """Returns whether the source content is a HEIC image."""
        return self.type(source) == "HEIC"

    def is_heif_family(self, source: str | bytes) -> bool:
        """Returns whether the source content is a HEIF-family image."""
        return self.type(source) in {"AVIF", "HEIC", "HEIF"}

    def _type_from_header(self, header: bytes) -> str | None:
        """Returns the image type from file header bytes."""
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "PNG"
        if header.startswith(b"\xff\xd8\xff"):
            return "JPEG"
        return self._heif_type(header)

    def _read_header(self, source: str | bytes) -> bytes:
        """Returns enough image container header bytes for metadata probing."""
        if isinstance(source, bytes):
            return source[:self._FTYP_HEADER_SIZE]
        with open(source, "rb") as file:
            return file.read(self._FTYP_HEADER_SIZE)

    @staticmethod
    def _openable_source(source: str | bytes) -> str | BytesIO:
        """Returns a path or in-memory stream suitable for image libraries."""
        return BytesIO(source) if isinstance(source, bytes) else source

    def _heif_type(self, header: bytes) -> str | None:
        """Returns the HEIF-family image type from an ISO BMFF ftyp box."""
        brands = self._ftyp_brands(header)
        if not brands:
            return None
        if brands & {b"avif", b"avis"}:
            return "AVIF"
        if brands & {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis"}:
            return "HEIC"
        if brands & {b"mif1", b"msf1"}:
            return "HEIF"
        return None

    def _ftyp_brands(self, header: bytes) -> set[bytes]:
        """Returns ISO BMFF major and compatible brands from an ftyp box."""
        if len(header) < 16 or header[4:8] != b"ftyp":
            return set()

        box_size = int.from_bytes(header[:4], "big")
        payload_offset = 8
        if box_size == 1:
            if len(header) < 24:
                return set()
            box_size = int.from_bytes(header[8:16], "big")
            payload_offset = 16
        elif box_size == 0:
            box_size = len(header)

        if box_size <= payload_offset or box_size > len(header):
            box_size = len(header)

        payload = header[payload_offset:box_size]
        if len(payload) < 8:
            return set()

        brands = {payload[:4]}
        compatible_brands = payload[8:]
        for index in range(0, len(compatible_brands) - 3, 4):
            brands.add(compatible_brands[index:index + 4])
        return brands


METADATA_READER = ImageMetadataReader()


class ImageDecoder:
    """Decodes one image through a named backend."""

    def __init__(self, name: str) -> None:
        self.name = name

    def hwaccel(self, source: Path) -> bool | None:
        """Returns hardware acceleration status when the backend can report it."""
        return None

    def read_metadata(self, source: Path) -> dict[str, Any]:
        """Returns image type and size without decoding pixels."""
        raise NotImplementedError

    def decode(self, source: Path) -> Image.Image:
        """Returns the decoded primary image as an RGB Pillow image."""
        raise NotImplementedError


class PillowDecoder(ImageDecoder):
    """Decodes images using Pillow or pillow-heif based on the source content."""

    def __init__(self) -> None:
        super().__init__("pillow")

    def decode(self, source: Path) -> Image.Image:
        """Loads the source with the selected backend and normalizes it to RGB."""
        if self._is_heif_source(source):
            return self._load_with_heif(source)
        return self._load_with_pillow(source)

    def read_metadata(self, source: Path) -> dict[str, Any]:
        """Returns image type and size using the selected backend."""
        return {
            "type": METADATA_READER.type(str(source)),
            "size": METADATA_READER.size(str(source)),
        }

    def _load_with_pillow(self, source: Path) -> Image.Image:
        """Loads a Pillow-supported image and normalizes it to RGB."""
        with Image.open(source) as image:
            image.load()
            return image.convert("RGB")

    def _load_with_heif(self, source: Path) -> Image.Image:
        """Loads a HEIF image through pillow-heif and normalizes it to RGB."""
        heif = open_heif(str(source), thumbnails=False)
        return Image.frombytes(
            heif.mode,
            heif.size,
            heif.data,
            "raw",
        ).convert("RGB")

    def _is_heif_source(self, source: Path) -> bool:
        """Returns whether the source content is a HEIF-family container."""
        return METADATA_READER.is_heif_family(str(source))


class TtheifDecoder(ImageDecoder):
    """Decodes HEIC images using libttheif."""

    def __init__(self, lib_path: str) -> None:
        super().__init__("ttheif")
        self.decoder = HeifDecoder(lib_path)

    def decode(self, source: Path) -> Image.Image:
        """Runs libttheif and normalizes the decoded image to RGB."""
        image = self.decoder.decode(str(source))
        if image is None:
            raise RuntimeError(f"libttheif failed to decode {source}")
        return image.convert("RGB")

    def read_metadata(self, source: Path) -> dict[str, Any]:
        """Returns HEIF image type and size without decoding pixels."""
        return {
            "type": METADATA_READER.type(str(source)),
            "size": METADATA_READER.size(str(source)),
        }


class AvDecoder(ImageDecoder):
    """Decodes HEIC primary streams or grid tiles using AV."""

    def __init__(self, name: str, accelerator: HWAccel | None) -> None:
        super().__init__(name)
        self.accelerator = accelerator

    def hwaccel(self, source: Path) -> bool | None:
        """Returns whether streams used for output are hardware accelerated."""
        if self.accelerator is None:
            return None
        return self._is_hardware_decode_enabled(source)

    def decode(self, source: Path) -> Image.Image:
        """Decodes a HEIC primary image or its grid tiles through AV."""
        image_size = self._get_heif_size(source)
        open_options = (
            {"hwaccel": self.accelerator} if self.accelerator is not None else {}
        )
        with av.open(str(source), **open_options) as container:
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
                        f"AV failed to decode the primary image stream for {image_size}"
                    ) from primary_decode_error
                raise RuntimeError(
                    f"AV found no image stream or grid tiles for {image_size}"
                )

            tile_size = self._stream_size(tiles[0])
            if tile_size is None:
                raise RuntimeError("AV exposed grid tiles without a valid tile size")
            tile_width, tile_height = tile_size
            columns = math.ceil(image_size[0] / tile_width)
            rows = math.ceil(image_size[1] / tile_height)
            required_tiles = columns * rows
            if len(tiles) < required_tiles:
                raise RuntimeError(
                    f"HEIF grid needs {required_tiles} tiles for {image_size}, "
                    f"but AV exposed {len(tiles)}"
                )

            image = Image.new("RGB", (columns * tile_width, rows * tile_height))
            for position, stream in enumerate(tiles[:required_tiles]):
                tile = self._decode_stream(container, stream)
                x = (position % columns) * tile_width
                y = (position // columns) * tile_height
                image.paste(tile, (x, y))
            return image.crop((0, 0, image_size[0], image_size[1]))

    def read_metadata(self, source: Path) -> dict[str, Any]:
        """Returns HEIF image type and size without decoding pixels."""
        return {
            "type": METADATA_READER.type(str(source)),
            "size": METADATA_READER.size(str(source)),
        }

    def _get_heif_size(self, source: Path) -> tuple[int, int]:
        """Returns the final HEIF presentation size."""
        return METADATA_READER.size(str(source))

    def _is_hardware_decode_enabled(self, source: Path) -> bool:
        """Reports whether all AV streams used for the output use hwaccel."""
        image_size = self._get_heif_size(source)
        with av.open(str(source), hwaccel=self.accelerator) as container:
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
        """Decodes one AV stream and copies its first frame to a host RGB image."""
        for frame in container.decode(stream):
            return frame.to_image().convert("RGB")
        raise RuntimeError(
            f"AV did not return an image frame for stream {stream.index}"
        )

    @staticmethod
    def _stream_size(stream: Any) -> tuple[int, int] | None:
        """Returns an AV stream's coded size when it is available."""
        context = stream.codec_context
        if context.width <= 0 or context.height <= 0:
            return None
        return (context.width, context.height)

    @staticmethod
    def _dependent_disposition(streams: list[Any]) -> int | None:
        """Returns AV's dependent stream disposition flag."""
        return type(streams[0].disposition).dependent if streams else None

    @staticmethod
    def _is_dependent_stream(stream: Any, dependent: int | None) -> bool:
        """Returns whether an AV stream is marked as a dependent item."""
        return dependent is not None and bool(stream.disposition & dependent)


@dataclass(frozen=True)
class Measurements:
    """Latency samples and source metadata for a benchmark run."""

    metadata_samples_ms: list[float]
    decode_samples_ms: list[float]
    image_type: str
    metadata_size: tuple[int, int]


def summarize_samples(samples_ms: list[float]) -> tuple[float, float]:
    """Returns mean and p95 latency for one sample set."""
    ordered = sorted(samples_ms)
    index_95 = math.ceil(len(ordered) * 0.95) - 1
    return statistics.mean(samples_ms), ordered[index_95]


def validate_matches_pillow(decoder: ImageDecoder, source: Path) -> None:
    """Validates that a decoder matches Pillow's decoded RGB output."""
    expected = PillowDecoder().decode(source)
    actual = decoder.decode(source)
    if actual.mode != expected.mode:
        raise RuntimeError(
            f"{decoder.name} returned image mode {actual.mode}, expected {expected.mode}"
        )
    if actual.size != expected.size:
        raise RuntimeError(
            f"{decoder.name} returned image size {actual.size}, expected {expected.size}"
        )
    difference = ImageChops.difference(actual, expected)
    extrema = difference.getextrema()
    max_difference = max(channel_extrema[1] for channel_extrema in extrema)
    mean_difference = max(ImageStat.Stat(difference).mean)
    if (
        max_difference > MAX_PIXEL_DIFFERENCE
        or mean_difference > MAX_MEAN_PIXEL_DIFFERENCE
    ):
        expected_path, actual_path = save_validation_images(
            decoder.name,
            source,
            expected,
            actual,
        )
        print(
            f"WARNING: {decoder.name} output differs from pillow: "
            f"max_delta={max_difference}, mean_delta={mean_difference:.3f}; "
            f"saved expected={expected_path} actual={actual_path}"
        )


def save_validation_images(
    decoder_name: str,
    source: Path,
    expected: Image.Image,
    actual: Image.Image,
) -> tuple[Path, Path]:
    """Saves Pillow and decoder outputs for visual comparison."""
    expected_path = source.with_name(f"{source.stem}.{GROUNDTRUTH_SUFFIX}.png")
    actual_path = source.with_name(f"{source.stem}.{decoder_name}.png")
    expected.save(expected_path)
    actual.save(actual_path)
    return expected_path, actual_path


def measure(
    decoder: ImageDecoder,
    source: Path,
    warmup: int,
    runs: int,
) -> Measurements:
    """Measures one provider after warmup iterations."""
    metadata = None
    for _ in range(warmup):
        metadata = decoder.read_metadata(source)
        decoder.decode(source)

    metadata_samples = []
    decode_samples = []
    gc.disable()
    try:
        for _ in range(runs):
            start_ns = time.perf_counter_ns()
            metadata = decoder.read_metadata(source)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            metadata_samples.append(elapsed_ms)

            start_ns = time.perf_counter_ns()
            decoder.decode(source)
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            decode_samples.append(elapsed_ms)
    finally:
        gc.enable()
    if metadata is None:
        raise RuntimeError("benchmark did not read image metadata")
    return Measurements(
        metadata_samples,
        decode_samples,
        metadata["type"],
        metadata["size"],
    )


def print_result(
    decoder: ImageDecoder,
    measurements: Measurements,
    hwaccel: bool | None,
) -> None:
    """Prints compact benchmark statistics in milliseconds."""
    metadata_mean_ms, metadata_p95_ms = summarize_samples(
        measurements.metadata_samples_ms
    )
    decode_mean_ms, decode_p95_ms = summarize_samples(measurements.decode_samples_ms)
    hwaccel_text = "-" if hwaccel is None else str(hwaccel).lower()
    metadata_size = f"{measurements.metadata_size[0]}x{measurements.metadata_size[1]}"
    print(
        "provider   type   metadata_size  hwaccel  runs  "
        "metadata_mean_ms  metadata_p95_ms  decode_mean_ms  decode_p95_ms"
    )
    print(
        f"{decoder.name:<10} {measurements.image_type:<6} {metadata_size:<13} "
        f"{hwaccel_text:<8} {len(measurements.decode_samples_ms):>4} "
        f"{metadata_mean_ms:>16.3f} "
        f"{metadata_p95_ms:>15.3f} "
        f"{decode_mean_ms:>14.3f} "
        f"{decode_p95_ms:>13.3f}"
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
        help="The decoder to use, e.g. pillow, av-sw, av-hw, or ttheif",
    )
    parser.add_argument(
        "--ttheif-lib",
        default=DEFAULT_TTHEIF_LIB,
        help=f"libttheif decoder library path (default: {DEFAULT_TTHEIF_LIB})",
    )
    parser.add_argument(
        "--device", default="cuda", help="AV HWAccel device type (default: cuda)"
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="warmup runs per provider (default: 3)"
    )
    parser.add_argument(
        "--runs", type=int, default=20, help="measured runs per provider (default: 20)"
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.runs <= 0:
        parser.error("--warmup must be non-negative and --runs must be positive")

    if args.tool == "pillow":
        decoder = PillowDecoder()
    elif args.tool == "ttheif":
        if not METADATA_READER.is_heic(str(args.image)):
            parser.error("--tool ttheif requires a HEIC image")
        decoder = TtheifDecoder(args.ttheif_lib)
    elif args.tool == "av-sw":
        if not METADATA_READER.is_heic(str(args.image)):
            parser.error("--tool av-sw requires a HEIC image")
        decoder = AvDecoder("av-sw", None)
    elif args.tool == "av-hw":
        if not METADATA_READER.is_heic(str(args.image)):
            parser.error("--tool av-hw requires a HEIC image")
        accelerator = HWAccel(device_type=args.device, allow_software_fallback=False)
        decoder = AvDecoder("av-hw", accelerator)
    else:
        parser.error(f"Unsupported tool: {args.tool}")

    if decoder.name != "pillow":
        validate_matches_pillow(decoder, args.image)

    measurements = measure(decoder, args.image, args.warmup, args.runs)
    hwaccel = decoder.hwaccel(args.image)
    print_result(decoder, measurements, hwaccel)
    if args.tool == "av-hw" and not hwaccel:
        print(
            "WARNING: av-hw did not report hardware-accelerated decoding for the selected image streams."
        )


if __name__ == "__main__":
    main()
