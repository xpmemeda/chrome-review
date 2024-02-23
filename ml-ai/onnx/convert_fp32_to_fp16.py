import sys
import onnx
from onnxmltools.utils import float16_converter


def main():
    src_model_path = sys.argv[1]
    dst_model_path = sys.argv[2]
    onnx_model = onnx.load_model(src_model_path)
    onnx_model = float16_converter.convert_float_to_float16(
        onnx_model, keep_io_types=True
    )
    onnx.save_model(onnx_model, dst_model_path)


if __name__ == "__main__":
    main()
