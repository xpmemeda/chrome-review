import argparse
import collections
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import subprocess
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel

parser = argparse.ArgumentParser(add_help=True)
subparsers = parser.add_subparsers(help="cmd")
subparsers.required = True


def _register_func(*funcs):
    def callback(impl):
        subparser = subparsers.add_parser(impl.__name__)
        subparser.set_defaults(func=impl)
        for func in funcs:
            func(subparser)

    return callback


def _add_model_path(parser):
    parser.add_argument("--model-path", required=True)


def _add_op_type(parser):
    parser.add_argument("--op-type", required=True)


@_register_func(_add_model_path)
def print_summary(args):
    saved_model = SavedModel()
    with open(os.path.join(args.model_path, "saved_model.pb"), "rb") as f:
        saved_model.ParseFromString(f.read())

    op2nums = collections.defaultdict(int)
    for meta_graph in saved_model.meta_graphs:
        for node in meta_graph.graph_def.node:
            op2nums[node.op] += 1
        for func in meta_graph.graph_def.library.function:
            for node in func.node_def:
                op2nums[node.op] += 1

    print("[operations]\n")
    for k, v in op2nums.items():
        print("{:<40s} {}".format(k, v))

    subprocess.run(["saved_model_cli", "show", "--all", "--dir", args.model_path])


@_register_func(_add_model_path)
def print_all_nodes(args):
    saved_model = SavedModel()
    with open(os.path.join(args.model_path, "saved_model.pb"), "rb") as f:
        saved_model.ParseFromString(f.read())
    for meta_graph in saved_model.meta_graphs:
        for node in meta_graph.graph_def.node:
            print(node)
            print("")


@_register_func(_add_model_path, _add_op_type)
def print_specific_nodes(args):
    saved_model = SavedModel()
    with open(os.path.join(args.model_path, "saved_model.pb"), "rb") as f:
        saved_model.ParseFromString(f.read())
    for meta_graph in saved_model.meta_graphs:
        for node in meta_graph.graph_def.node:
            if node.op == args.op_type:
                print(node)
                print("")


@_register_func(_add_model_path, _add_op_type)
def print_node_inputs(args):
    saved_model = SavedModel()
    with open(os.path.join(args.model_path, "saved_model.pb"), "rb") as f:
        saved_model.ParseFromString(f.read())
    nodes_map = {}
    for meta_graph in saved_model.meta_graphs:
        for node in meta_graph.graph_def.node:
            nodes_map[node.name] = node
            if node.op == args.op_type:
                print(node)
                for i in node.input:
                    print(nodes_map[i])
                print("-" * 100)


if __name__ == "__main__":
    try:
        args = parser.parse_args()
    except TypeError:
        parser.print_usage()
        exit()
    args.func(args)
