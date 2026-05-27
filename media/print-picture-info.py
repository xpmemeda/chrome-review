import argparse
import json
import mimetypes

from pathlib import Path
from PIL import ExifTags, Image
from pillow_heif import register_heif_opener

HEIF_SUFFIXES = {".heic", ".heif", ".hif"}
BINARY_METADATA_KEYS = {"exif", "icc_profile"}


def json_value(value: object) -> object:
    """Converts arbitrary Pillow metadata into a value suitable for JSON output."""
    if isinstance(value, bytes):
        return {"bytes": len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return str(value)


def read_exif(image: Image.Image) -> dict[str, object]:
    """Returns EXIF tags with readable names and JSON-safe values."""
    exif = image.getexif()
    return {
        ExifTags.TAGS.get(tag_id, str(tag_id)): json_value(value)
        for tag_id, value in exif.items()
    }


def read_metadata(image: Image.Image) -> dict[str, object]:
    """Returns image metadata while representing large binary payloads compactly."""
    metadata = {}
    for key, value in image.info.items():
        if key in BINARY_METADATA_KEYS and isinstance(value, bytes):
            metadata[key] = {"bytes": len(value)}
        else:
            metadata[key] = json_value(value)
    return metadata


def inspect_image(source: Path) -> dict[str, object]:
    """Opens one image and returns commonly useful container and image properties."""
    with Image.open(source) as image:
        image.load()
        mime_type = Image.MIME.get(image.format or "")
        if mime_type is None:
            mime_type = mimetypes.guess_type(source.name)[0]
        return {
            "path": str(source),
            "file_size_bytes": source.stat().st_size,
            "format": image.format,
            "mime_type": mime_type,
            "size": {"width": image.width, "height": image.height},
            "mode": image.mode,
            "bands": list(image.getbands()),
            "frames": getattr(image, "n_frames", 1),
            "animated": bool(getattr(image, "is_animated", False)),
            "metadata": read_metadata(image),
            "exif": read_exif(image),
        }


def human_size(byte_count: int) -> str:
    """Formats an integer byte count using binary size units."""
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{byte_count} B"


def format_metadata_value(value: object) -> str:
    """Formats one metadata value for compact terminal display."""
    if isinstance(value, dict) and set(value) == {"bytes"}:
        return human_size(int(value["bytes"]))
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def print_image_info(info: dict[str, object]) -> None:
    """Prints an image information record as readable text."""
    size = info["size"]
    assert isinstance(size, dict)
    file_size = info["file_size_bytes"]
    assert isinstance(file_size, int)
    print(f"File:       {info['path']}")
    print(f"File size:  {human_size(file_size)} ({file_size} bytes)")
    print(f"Format:     {info['format'] or 'unknown'}")
    print(f"MIME type:  {info['mime_type'] or 'unknown'}")
    print(f"Dimensions: {size['width']} x {size['height']}")
    print(f"Mode:       {info['mode']} ({', '.join(info['bands'])})")
    print(f"Frames:     {info['frames']} (animated: {str(info['animated']).lower()})")

    metadata = info["metadata"]
    assert isinstance(metadata, dict)
    if metadata:
        print("Metadata:")
        for key in sorted(metadata):
            print(f"  {key}: {format_metadata_value(metadata[key])}")

    exif = info["exif"]
    assert isinstance(exif, dict)
    if exif:
        print("EXIF:")
        for key in sorted(exif):
            print(f"  {key}: {format_metadata_value(exif[key])}")


def main() -> None:
    """Parses CLI arguments and prints information for each supplied image."""
    parser = argparse.ArgumentParser(
        description="Print information for PNG, JPEG, HEIC/HEIF, and other Pillow-supported images."
    )
    parser.add_argument(
        "--sources", type=Path, nargs="+", help="image files to inspect"
    )
    parser.add_argument("--json", action="store_true", help="print records as JSON")
    args = parser.parse_args()

    # NOTE: thumbnails=False 表示关闭 HEIF/HEIC 文件中内嵌缩略图的支持
    register_heif_opener(thumbnails=False)

    records = []
    for source in args.sources:
        records.append(inspect_image(source))

    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False))
        return
    for index, record in enumerate(records):
        if index:
            print()
        print_image_info(record)


if __name__ == "__main__":
    main()
