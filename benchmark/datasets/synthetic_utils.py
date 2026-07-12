import struct
import typing as ty
import zlib


def check_prompt_prefix_hit_rate(prompt_prefix_hit_rate: float) -> None:
    if prompt_prefix_hit_rate < 0.0 or prompt_prefix_hit_rate > 1.0:
        raise RuntimeError("prompt_prefix_hit_rate must be in [0, 1].")


def make_synthetic_system_prompt(num_prompt_prefix_tokens: int) -> ty.Optional[str]:
    if not num_prompt_prefix_tokens:
        return None
    return " hi" * num_prompt_prefix_tokens


def make_synthetic_prompt(
    req_idx: int,
    seed: int,
    num_prompt_suffix_tokens: int,
) -> str:
    random_txt = str(seed + req_idx)
    instruction_txt = make_synthetic_output_instruction()
    filler_txt = " hi" * max(
        0,
        num_prompt_suffix_tokens
        - len(random_txt.split())
        - len(instruction_txt.split()),
    )
    return random_txt + filler_txt + ". " + instruction_txt


def make_synthetic_output_instruction() -> str:
    return "Tell me a long story."


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind)
    crc = zlib.crc32(data, crc)
    return (
        struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc & 0xFFFFFFFF)
    )


def make_png(width: int, height: int, seed: int) -> bytes:
    rows = make_base_rgb_rows(width, height, seed)
    return encode_png_rgb(width, height, rows)


def make_base_rgb_rows(width: int, height: int, seed: int) -> bytearray:
    palette = [
        (
            (seed * 17 + 211) & 0xFF,
            (seed * 29 + 223) & 0xFF,
            (seed * 31 + 229) & 0xFF,
        ),
        (
            (seed * 37 + 180) & 0xFF,
            (seed * 41 + 205) & 0xFF,
            (seed * 43 + 232) & 0xFF,
        ),
        (
            (seed * 47 + 236) & 0xFF,
            (seed * 53 + 216) & 0xFF,
            (seed * 59 + 188) & 0xFF,
        ),
    ]
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        row_color = palette[(y // 128) % len(palette)]
        for x in range(width):
            color = row_color
            if width // 8 <= x < width * 3 // 8 and height // 7 <= y < height // 4:
                color = (76, 116, 178)
            elif (
                width * 5 // 8 <= x < width * 7 // 8
                and height // 3 <= y < height * 10 // 21
            ):
                color = (214, 139, 76)
            elif (
                width // 4 <= x < width * 3 // 4
                and height * 5 // 7 <= y < height * 6 // 7
            ):
                color = (88, 154, 116)
            rows.extend(color)
    return rows


def encode_png_rgb(width: int, height: int, rows: ty.Union[bytes, bytearray]) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=1))
        + _png_chunk(b"IEND", b"")
    )
