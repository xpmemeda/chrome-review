import pdb
import sys
import tensorflow.compat.v1 as tf


def extract_tensors(signature_def, graph):
    output = dict()

    for key in signature_def:
        value = signature_def[key]

        if isinstance(value, tf.TensorInfo):
            output[key] = graph.get_tensor_by_name(value.name)

    return output


def extract_tags(signature_def, graph):
    output = dict()

    for key in signature_def:
        output[key] = dict()
        output[key]["inputs"] = extract_tensors(signature_def[key].inputs, graph)
        output[key]["outputs"] = extract_tensors(signature_def[key].outputs, graph)

    return output


# saved_model_cli show --all --dir <saved_model/path>

with tf.Session(graph=tf.Graph()) as session:
    serve = tf.saved_model.load(session, tags=["serve"], export_dir=sys.argv[1])

    tags = extract_tags(serve.signature_def, session.graph)
    model = tags["serving_default"]
    pdb.set_trace()
    exit()
