import sys
import os
import subprocess
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel


def main():
    saved_model = SavedModel()
    with open(os.path.join(sys.argv[1], "saved_model.pb"), "rb") as f:
        saved_model.ParseFromString(f.read())
    model_op_names = set()
    # Iterate over every metagraph in case there is more than one
    for meta_graph in saved_model.meta_graphs:
        # Add operations in the graph definition
        model_op_names.update(node.op for node in meta_graph.graph_def.node)
        # Go through the functions in the graph definition
        for func in meta_graph.graph_def.library.function:
            # Add operations in each function
            model_op_names.update(node.op for node in func.node_def)
    # Convert to list, sorted if you want
    model_op_names = sorted(model_op_names)
    print(*model_op_names, sep="\n")

    subprocess.run(["saved_model_cli", "show", "--all", "--dir", sys.argv[1]])


if __name__ == "__main__":
    main()
