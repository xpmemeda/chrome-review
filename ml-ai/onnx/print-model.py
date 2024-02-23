import onnx
import sys
import argparse


parser = argparse.ArgumentParser(add_help=True)
subparsers = parser.add_subparsers(help="cmd")
subparsers.required = True

onnxelementtype2string = {
    0: "undefined",
    1: "float",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "double",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}


def __register_func(*funcs):
    def callback(impl):
        subparser = subparsers.add_parser(impl.__name__)
        subparser.set_defaults(func=impl)
        for func in funcs:
            func(subparser)

    return callback


def __model_path(parser):
    parser.add_argument("--model", required=True)


@__register_func(__model_path)
def nodes(args):
    model = onnx.load(args.model)
    for node in model.graph.node:
        print("{}|{}".format(node.op_type, node.name))
        print("{}".format(node.input))
        print("{}".format(node.output))
        print()


@__register_func(__model_path)
def io(args):
    model = onnx.load(args.model)

    def get_shape(x):
        r = []
        for v in x.type.tensor_type.shape.dim:
            if v.dim_param:
                r.append(-1)
            else:
                r.append(v.dim_value)
        return r

    print("[Ins]")
    for x in model.graph.input:
        name = x.name
        type_str = onnxelementtype2string[x.type.tensor_type.elem_type]
        shape_list = get_shape(x)
        print("{:<40}{:<10}{!r:<30}".format(name, type_str, shape_list))
    print("\n[Outs]")
    for x in model.graph.output:
        name = x.name
        type_str = onnxelementtype2string[x.type.tensor_type.elem_type]
        shape_list = get_shape(x)
        print("{:<40}{:<10}{!r:<30}".format(name, type_str, shape_list))


if __name__ == "__main__":
    try:
        args = parser.parse_args()
    except TypeError:
        parser.print_usage()
        sys.exit(1)
    args.func(args)
