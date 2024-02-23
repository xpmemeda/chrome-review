import onnx
from onnx import numpy_helper
import sys
import numpy as np

def replace_zip_map(graph, zip_map_node):
    # Find output to be replaced
    replaced_out = None
    replaced_idx = 0
    for j in range(len(graph.output)):
        if graph.output[j].name == zip_map_node.output[0]:
            replaced_out = graph.output[j]
            replaced_idx = j
            break
    if replaced_out is None:
        print("UnKnown ERR")
        exit()

    # Modify graph
    graph.node.remove(zip_map_node)
    graph.output.remove(replaced_out)
    zip_map_key, zip_map_value = zip_map_node.output[0] + ":key", zip_map_node.output[0] + ":value"
    squeeze_node = onnx.helper.make_node('Squeeze', inputs=zip_map_node.input, outputs=[zip_map_value])
    graph.node.append(squeeze_node)
    len_zip_map = zip_map_node.attribute[0].ints.__len__()
    graph.output.insert(replaced_idx, onnx.helper.make_tensor_value_info(zip_map_value, onnx.TensorProto.FLOAT, (len_zip_map,)))
    graph.output.insert(replaced_idx, onnx.helper.make_tensor_value_info(zip_map_key, onnx.TensorProto.INT64, (len_zip_map,)))
    key = onnx.numpy_helper.from_array(np.array(zip_map_node.attribute[0].ints), zip_map_key)
    graph.initializer.append(key)

    return

def main():
    if len(sys.argv) != 3:
        print("Usage: python replace_zip_map.py <old_model> <new_model>")
        return -1
    model = onnx.load(sys.argv[1])
    graph = model.graph
    i = 0
    while i < len(graph.node):
        if graph.node[i].op_type != 'ZipMap':
            i += 1
            continue
        print("Replace ZipMap: index = {}".format(i))
        replace_zip_map(graph, graph.node[i])

    onnx.save(model, sys.argv[2])

if __name__ == '__main__':
    main()
