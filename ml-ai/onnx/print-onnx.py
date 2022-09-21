import onnx
import sys

"""
    UNDEFINED = 0;
    FLOAT = 1;
    INT = 2;
    STRING = 3;
    TENSOR = 4;
    GRAPH = 5;
    SPARSE_TENSOR = 11;
    TYPE_PROTO = 13;

    FLOATS = 6;
    INTS = 7;
    STRINGS = 8;
    TENSORS = 9;
    GRAPHS = 10;
    SPARSE_TENSORS = 12;
    TYPE_PROTOS = 14;
"""


def main():
    model = onnx.load(sys.argv[1])
    output = [node.name for node in model.graph.output]

    input_all = [node.name for node in model.graph.input]
    input_initializer = [node.name for node in model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))

    input_node = []
    for node in model.graph.input:
        if node.name in net_feed_input:
            input_node.append(node)
    print(input_node)
    print("Inputs: ", net_feed_input)
    print("Outputs: ", output)


if __name__ == "__main__":
    main()
