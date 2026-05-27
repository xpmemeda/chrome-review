import ctypes
import os
from typing import Optional

from PIL import Image


class Rect(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint32),
        ("y", ctypes.c_uint32),
        ("w", ctypes.c_uint32),
        ("h", ctypes.c_uint32),
    ]


class HeifDecodingParam(ctypes.Structure):
    _fields_ = [
        ("in_sample", ctypes.c_float),
        ("use_wpp", ctypes.c_bool),
        ("threads", ctypes.c_uint32),
        ("decode_rect", ctypes.c_bool),
        ("rect", Rect),
        ("use_extern_buffer", ctypes.c_bool),
        ("in_sample_mode", ctypes.c_int),
    ]


class HeifColrInfo(ctypes.Structure):
    _fields_ = [
        ("color_type", ctypes.c_uint32),
        ("color_primaries", ctypes.c_uint16),
        ("transfer_characteristics", ctypes.c_uint16),
        ("matrix_coefficients", ctypes.c_uint16),
        ("full_range_flag", ctypes.c_uint8),
    ]


class HeifOutputStream(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint32),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("exif_size", ctypes.c_uint32),
        ("exif_data", ctypes.POINTER(ctypes.c_uint8)),
        ("duration_table", ctypes.POINTER(ctypes.c_uint64)),
        ("frame_num", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("duration", ctypes.c_uint64),
        ("single_duration", ctypes.c_uint64),
        ("bit_depth", ctypes.c_uint8),
        ("icc_data", ctypes.POINTER(ctypes.c_uint8)),
        ("icc_size", ctypes.c_uint32),
        ("pix_fmt", ctypes.c_uint8),
        ("colr_info", HeifColrInfo),
        ("error_code", ctypes.c_int),
    ]


class HeifDecoder:
    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(os.path.abspath(lib_path))
        self.lib.heif_decode_to_rgb.restype = HeifOutputStream
        self.lib.heif_release_output_stream.argtypes = [
            ctypes.POINTER(HeifOutputStream)
        ]

    def decode(self, heif_path: str) -> Optional[Image.Image]:
        with open(heif_path, "rb") as f:
            heif_data = f.read()

        data_ptr = ctypes.cast(
            ctypes.create_string_buffer(heif_data), ctypes.POINTER(ctypes.c_uint8)
        )
        data_size = ctypes.c_uint32(len(heif_data))

        param = HeifDecodingParam()
        param.in_sample = 1.0
        param.use_wpp = True
        param.threads = 2
        param.decode_rect = False
        param.use_extern_buffer = True
        param.in_sample_mode = 0  # SampleModeLegacy

        output = self.lib.heif_decode_to_rgb(data_ptr, data_size, ctypes.byref(param))

        if output.size > 0 and bool(output.data):
            print(
                f"[INFO] Decoded image: {output.width}x{output.height}, depth={output.bit_depth}"
            )
            try:
                buf_type = ctypes.c_uint8 * output.size
                raw_buf = ctypes.cast(output.data, ctypes.POINTER(buf_type)).contents
                rgb_data = bytes(raw_buf)
                return Image.frombytes("RGB", (output.width, output.height), rgb_data)
            finally:
                self.lib.heif_release_output_stream(ctypes.byref(output))
        else:
            print(f"[ERROR] Decoding failed with error code: {output.error_code}")
            return None
