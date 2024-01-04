import sys
import argparse
import os
import tensorflow as tf

parser = argparse.ArgumentParser(add_help=True)
subparsers = parser.add_subparsers(help="cmd")
subparsers.required = True


def __register_func(*funcs):
    def callback(impl):
        subparser = subparsers.add_parser(impl.__name__)
        subparser.set_defaults(func=impl)
        for func in funcs:
            func(subparser)

    return callback


def __model_path(parser):
    parser.add_argument("--ckpt-dir", required=True)


def get_model_name(source_ckpt_dir):
    meta_file = None
    for file_name in os.listdir(source_ckpt_dir):
        if file_name.endswith("meta"):
            meta_file = file_name
            break
    assert meta_file
    return meta_file[:-5]


@__register_func(__model_path)
def print_all_nodes(args):
    source_ckpt_dir = args.ckpt_dir

    model_name = get_model_name(source_ckpt_dir)

    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(
            os.path.join(source_ckpt_dir, model_name + ".meta")
        )
        loader.restore(sess, os.path.join(source_ckpt_dir, model_name))
        print(sess.graph_def)


@__register_func(__model_path)
def print_inputs_and_outputs(args):
    source_ckpt_dir = args.ckpt_dir

    model_name = get_model_name(source_ckpt_dir)

    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(
            os.path.join(source_ckpt_dir, model_name + ".meta")
        )
        loader.restore(sess, os.path.join(source_ckpt_dir, model_name))

        graph_def = graph.as_graph_def()

        operation_output_list = []
        operation_input_list = []
        for node in graph_def.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
            operation_output_list.append(node.name)
            operation_input_list.extend(node.input)
        unused_operation_outputs = set(operation_output_list) - set(
            operation_input_list
        )

        placeholders = []
        for node in graph_def.node:
            if node.op == "Placeholder":
                placeholders.append(node.name)

        graph_result_from_placeholders = set(placeholders)
        for node in graph_def.node:
            for x in node.input:
                if x in graph_result_from_placeholders:
                    graph_result_from_placeholders.add(node.name)
                    break

        outputs = graph_result_from_placeholders.intersection(unused_operation_outputs)

        print("inputs:", placeholders)
        print("outputs:", outputs)


if __name__ == "__main__":
    try:
        args = parser.parse_args()
    except TypeError:
        parser.print_usage()
        sys.exit(1)
    args.func(args)
