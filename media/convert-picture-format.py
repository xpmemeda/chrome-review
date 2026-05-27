import argparse

from pathlib import Path
from PIL import Image, ImageColor
from pillow_heif import register_heif_opener

OUTPUT_FORMATS = {
    ".heic": "HEIF",
    ".heif": "HEIF",
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
}
INPUT_FORMATS = {"HEIF", "JPEG", "PNG"}
METADATA_KEYS = ("exif", "icc_profile", "xmp")


def output_format(destination: Path) -> str:
    """Returns the Pillow output format selected by the destination suffix."""
    try:
        return OUTPUT_FORMATS[destination.suffix.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(OUTPUT_FORMATS))
        raise ValueError(
            f"unsupported destination suffix '{destination.suffix}'; use one of: {supported}"
        ) from exc


def copy_metadata(image: Image.Image) -> dict[str, object]:
    """Returns metadata that Pillow and pillow-heif can preserve while saving."""
    return {
        key: image.info[key] for key in METADATA_KEYS if image.info.get(key) is not None
    }


def flatten_transparency(image: Image.Image, background: str) -> Image.Image:
    """Returns an RGB image composited on the requested JPEG background color."""
    rgba_image = image.convert("RGBA")
    rgb_background = Image.new("RGB", rgba_image.size, ImageColor.getrgb(background))
    rgb_background.paste(rgba_image, mask=rgba_image.getchannel("A"))
    return rgb_background


def prepare_for_save(
    image: Image.Image, file_format: str, background: str
) -> Image.Image:
    """Converts unsupported modes where required by the selected encoder."""
    if file_format == "JPEG":
        has_transparency = "A" in image.getbands() or "transparency" in image.info
        if has_transparency:
            return flatten_transparency(image, background)
        if image.mode not in ("L", "RGB", "CMYK"):
            return image.convert("RGB")
    elif file_format == "HEIF" and image.mode not in ("RGB", "RGBA"):
        return image.convert("RGBA" if "A" in image.getbands() else "RGB")
    return image


def convert_image(
    source: Path,
    destination: Path,
    quality: int,
    background: str,
) -> None:
    """Reads a supported image and saves it in the destination image format."""
    file_format = output_format(destination)
    with Image.open(source) as source_image:
        if source_image.format not in INPUT_FORMATS:
            supported = ", ".join(sorted(INPUT_FORMATS))
            raise ValueError(
                f"unsupported input format '{source_image.format}'; supported formats: {supported}"
            )
        source_image.load()
        save_options = copy_metadata(source_image)
        output_image = prepare_for_save(source_image, file_format, background)
        if file_format in ("JPEG", "HEIF"):
            save_options["quality"] = quality
        output_image.save(destination, format=file_format, **save_options)


def main() -> None:
    """Parses CLI arguments and converts an image."""
    parser = argparse.ArgumentParser(
        description="Convert images among PNG, JPEG, and HEIC/HEIF formats."
    )
    parser.add_argument(
        "--source", type=Path, help="input PNG, JPEG, or HEIC/HEIF file", required=True
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="output file; extension selects .png, .jpg/.jpeg, or .heic/.heif",
        required=True,
    )
    # NOTE: 相同的 quality 参数对于不同的压缩算法，其效果是不相同的。
    # 如果要使得 HEIC 图片的大小为 JPEG 图片大小的一半左右，可以把 quality 设置为 60 左右。JPEG 目标则保持为 95。
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG/HEIC output quality from 0 to 100 (default: 95)",
    )
    parser.add_argument(
        "--background",
        default="white",
        help="background color used when saving transparent images as JPEG (default: white)",
    )
    args = parser.parse_args()
    if not 0 <= args.quality <= 100:
        parser.error("--quality must be between 0 and 100")

    ImageColor.getrgb(args.background)
    register_heif_opener(thumbnails=False)
    convert_image(args.source, args.destination, args.quality, args.background)


if __name__ == "__main__":
    main()
