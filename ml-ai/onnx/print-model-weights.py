import argparse
import numpy
import onnx


def model_weights(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    weights_and_constants = {}
    for node in graph.node:
        if node.op_type != "Constant":
            continue

        constant_pair = {
            node.output[0]: onnx.numpy_helper.to_array(attr.t)
            for attr in node.attribute
            if attr.name == "value"
        }
        weights_and_constants.update(constant_pair)

    for initializer in graph.initializer:
        weight_name = initializer.name
        weight_value = onnx.numpy_helper.to_array(initializer)
        weights_and_constants[weight_name] = weight_value

    return weights_and_constants


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="onnx model path", required=True)
    parser.add_argument("--npz-path", type=str, help="npz path to save", required=True)
    arguments = parser.parse_args()

    model_path = arguments.model
    npz_path = arguments.npz_path

    w = model_weights(model_path)
    numpy.savez(npz_path, **w)
