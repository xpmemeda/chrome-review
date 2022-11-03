import tensorflow.compat.v1 as tf
import numpy as np
import argparse


def run_savedmodel(args):
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], args.model_path)

        npz_data = np.load(args.ins_file)
        ins_numpy = {in_name + ":0": npz_data[in_name] for in_name in npz_data.files}

        graph = tf.get_default_graph()
        out_names = args.outs
        outs = [graph.get_tensor_by_name(out_name) for out_name in out_names]
        print(sess.run(outs, feed_dict=ins_numpy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ins-file", required=True)
    parser.add_argument("--outs", nargs="+", type=str)
    args = parser.parse_args()
    run_savedmodel(args)
