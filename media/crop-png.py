import argparse

from pathlib import Path
from PIL import Image


def parse_size(value: str) -> tuple[int, int]:
    """Parses a crop size in WIDTHxHEIGHT form."""
    try:
        width, height = (int(component) for component in value.lower().split("x", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "size must be formatted as WIDTHxHEIGHT"
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def crop_image(
    source: Path,
    destination: Path,
    crop_size: tuple[int, int],
    left: int,
    top: int,
) -> None:
    """Crops the requested image rectangle and writes it to the output path."""
    with Image.open(source) as image:
        width, height = crop_size
        right = left + width
        bottom = top + height
        if left < 0 or top < 0 or right > image.width or bottom > image.height:
            raise ValueError(
                f"crop box ({left}, {top}, {right}, {bottom}) exceeds "
                f"image bounds ({image.width}, {image.height})"
            )
        image.crop((left, top, right, bottom)).save(destination, format="PNG")


def main() -> None:
    """Parses CLI arguments and crops an image."""
    parser = argparse.ArgumentParser(
        description="Crop an image and save the result as PNG."
    )
    parser.add_argument("--source", type=Path, required=True, help="input image path")
    parser.add_argument(
        "--destination", type=Path, required=True, help="output PNG path"
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        required=True,
        metavar="WIDTHxHEIGHT",
        help="cropped output size, for example 400x300",
    )
    parser.add_argument(
        "--left", type=int, default=0, help="left crop offset (default: 0)"
    )
    parser.add_argument(
        "--top", type=int, default=0, help="top crop offset (default: 0)"
    )
    args = parser.parse_args()

    crop_image(args.source, args.destination, args.size, args.left, args.top)


if __name__ == "__main__":
    main()
