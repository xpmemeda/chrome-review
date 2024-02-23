import sys
import onnx
from onnx import numpy_helper


def main():
    model_path = sys.argv[1]
    model = onnx.load(model_path)

    extra_output = []
    for name in ['2802', '2793', '2804', '2805']:
        intermediate_tensor_name = name  # (1, 1, 5)
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = intermediate_tensor_name
        extra_output.append(intermediate_layer_value_info)

    model.graph.output.extend(extra_output)
    onnx.save(model, model_path + ".debug.onnx")

if __name__ == "__main__":
    main()
