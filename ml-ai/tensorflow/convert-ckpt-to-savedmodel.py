import argparse
import sys
import copy
import os
import tensorflow as tf


def get_model_name(source_ckpt_dir):
    meta_file = None
    for file_name in os.listdir(source_ckpt_dir):
        if file_name.endswith("meta"):
            meta_file = file_name
            break
    assert meta_file
    return meta_file[:-5]


def get_default_inputs_and_outputs(graph_def):
    operation_output_list = []
    operation_input_list = []
    for node in graph_def.node:
        operation_output_list.append(node.name)
        operation_input_list.extend(node.input)
    unused_operation_outputs = set(operation_output_list) - set(operation_input_list)

    placeholders = set()
    for node in graph_def.node:
        if node.op == "Placeholder":
            placeholders.add(node.name)

    graph_result_from_placeholders = copy.deepcopy(placeholders)
    for node in graph_def.node:
        for x in node.input:
            if x in graph_result_from_placeholders:
                graph_result_from_placeholders.add(node.name)
                break

    outputs = graph_result_from_placeholders.intersection(unused_operation_outputs)

    print("warning: default inputs and outputs")
    print(placeholders)
    print(outputs)

    return placeholders, outputs


def main(args):
    source_ckpt_dir = args.ckpt_dir
    target_savedmodel_dir = args.output_path

    model_name = get_model_name(source_ckpt_dir)

    graph = tf.Graph()
    with tf.compat.v1.Session(graph=graph) as sess:
        # Restore from checkpoint
        loader = tf.compat.v1.train.import_meta_graph(
            os.path.join(source_ckpt_dir, model_name + ".meta")
        )
        loader.restore(sess, os.path.join(source_ckpt_dir, model_name))

        inputs, outputs = get_default_inputs_and_outputs(graph.as_graph_def())
        outputs = args.output_tensors if args.output_tensors else outputs

        input_tensors = {x: graph.get_tensor_by_name(x + ":0") for x in inputs}
        output_tensors = {x: graph.get_tensor_by_name(x + ":0") for x in outputs}

        from tensorflow.python.saved_model.signature_def_utils_impl import (
            predict_signature_def,
        )

        signature_def = predict_signature_def(input_tensors, output_tensors)

        # Export checkpoint to SavedModel
        builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(
            target_savedmodel_dir
        )
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.SERVING],
            signature_def_map={"serving_default": signature_def},
            strip_default_attrs=True,
        )
        builder.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--ckpt-dir", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--output-tensors", nargs="+", type=str)
    args = parser.parse_args()
    main(args)
