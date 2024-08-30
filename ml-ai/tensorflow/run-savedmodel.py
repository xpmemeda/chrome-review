import tensorflow.compat.v1 as tf
import numpy as np
import argparse


def print_numpy_array(arr):
    arr = arr.reshape([-1]).tolist()[:6]
    arr_str_format = len(arr) * "{:.2f}, "
    arr_str = arr_str_format.format(*arr)
    print("%s ..." % arr_str[:-2])
    print(
        "Min: %.2f, Max: %.2f, Avg: %.2f, Var: %.2f"
        % (np.min(arr), np.max(arr), np.mean(arr), np.var(arr))
    )


def run_savedmodel(args):
    with tf.Session(graph=tf.Graph()) as sess:
        metagraph = tf.saved_model.loader.load(sess, ["serve"], args.model_path)
        outputs_mapping = dict(metagraph.signature_def["serving_default"].outputs)

        out_names = (
            [v.name for k, v in outputs_mapping.items()] if not args.outs else args.outs
        )

        npz_data = np.load(args.ins_file)
        numpy_ins = {in_name + ":0": npz_data[in_name] for in_name in npz_data.files}

        graph = tf.get_default_graph()
        outs = [graph.get_tensor_by_name(out_name) for out_name in out_names]
        numpy_outs = sess.run(outs, feed_dict=numpy_ins)
        for out_name, numpy_out in zip(out_names, numpy_outs):
            print("\033[92m[%s]\033[0m" % out_name)
            try:
                print_numpy_array(numpy_out)
            except:
                print(numpy_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ins-file", required=True, help="Numpy Npz File")
    parser.add_argument("--outs", nargs="+", type=str)
    args = parser.parse_args()
    run_savedmodel(args)
