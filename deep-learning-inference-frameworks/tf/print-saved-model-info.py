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
def print_io_info(args):
    savedmodelcli_cmd = [
        "saved_model_cli",
        "show",
        "--signature_def",
        "serving_default",
        "--tag_set",
        "serve",
        "--dir",
        args.model_path,
    ]
    process = subprocess.Popen(
        savedmodelcli_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = process.communicate()
    errcode = process.returncode
    if errcode:
        print(err)
        return

    import re

    pattern_inp_name = re.compile(r"""inputs\['\S+'\]""")
    inp_names = pattern_inp_name.findall(str(out))
    inp_names = [name[8:-2] for name in inp_names]
    pattern_oup_name = re.compile(r"""outputs\['\S+'\]""")
    oup_names = pattern_oup_name.findall(str(out))
    oup_names = [name[9:-2] for name in oup_names]

    pattern_dtype = re.compile(r"""dtype: \S+\\n""")
    dtypes = pattern_dtype.findall(str(out))
    dtypes = [dtype[7:-2] for dtype in dtypes]

    pattern_shape = re.compile(r"""shape: \(.+?\)""")
    shapes = pattern_shape.findall(str(out))
    shapes = [list(map(int, shape[8:-1].split(", "))) for shape in shapes]

    assert len(inp_names) + len(oup_names) == len(dtypes) == len(shapes)
    print("[inp]")
    for name, dtype, shape in zip(inp_names, dtypes, shapes):
        print("{:<30s} dtype={:<20s}shape={}".format(name, dtype, shape))
    print("[oup]")
    for name, dtype, shape in zip(
        oup_names, dtypes[len(inp_names) :], shapes[len(inp_names) :]
    ):
        print("{:<30s} dtype={:<20s}shape={}".format(name, dtype, shape))


@_register_func(_add_model_path)
def count_nodes(args):
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

    print("[operations]")
    for k, v in op2nums.items():
        print("{:<40s} {}".format(k, v))


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
