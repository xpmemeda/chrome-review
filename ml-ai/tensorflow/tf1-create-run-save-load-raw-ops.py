import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()  # need to disable eager in TF2.x

import shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser(add_help=True)
subparsers = parser.add_subparsers(help="cmd")
subparsers.required = True


def _register_func(impl):
    subparser = subparsers.add_parser(impl.__name__)
    subparser.set_defaults(func=impl)


def rmdir(dirname):
    try:
        shutil.rmtree(dirname)
    except:
        pass

    return


@_register_func
def stringsplit():
    x = tf.placeholder(tf.string, [2])
    delimiter = tf.constant(" ", tf.string)
    stringsplit_op = tf.raw_ops.StringSplit(
        input=x, delimiter=delimiter, skip_empty=True
    )
    y = tf.sparse.to_dense(
        tf.SparseTensor(
            stringsplit_op.indices, stringsplit_op.values, stringsplit_op.shape
        )
    )

    dirname = "stringsplit"
    rmdir(dirname)
    with tf.Session() as sess:
        tf.saved_model.simple_save(sess, dirname, inputs={"x": x}, outputs={"y": y})
        print(
            sess.run(
                [y], feed_dict={"Placeholder:0": np.array(["hello world", "a b c"])}
            )
        )


@_register_func
def stringtohashbucketfast():
    tf.reset_default_graph()
    x = tf.placeholder(tf.string, shape=[2, 4])
    y = tf.raw_ops.StringToHashBucketFast(input=x, num_buckets=100)

    dirname = "stringtohashbucketfast"
    rmdir(dirname)
    x_numpy = np.array(
        [
            ["a_X_1_X_+", "a_X_1_X_-", "", ""],
            ["b_X_2_X_*", "b_X_2_X_/", "c_X_2_X_*", "c_X_2_X_/"],
        ],
        dtype="<S9",
    )
    np.savez(dirname + ".npz", x=x_numpy)
    with tf.Session() as sess:
        tf.saved_model.simple_save(sess, dirname, inputs={"x": x}, outputs={"y": y})
        print(
            sess.run(
                [y],
                feed_dict={"Placeholder:0": x_numpy},
            )
        )


@_register_func
def lookuptablefindv2():
    tf.reset_default_graph()
    keys = tf.constant(["a", "b", "c", "d"])
    values = tf.constant([1, 2, 3, 4])
    table_handle = tf.raw_ops.HashTableV2(
        key_dtype=tf.string, value_dtype=tf.int32, shared_name="run-raw-ops"
    )
    op = tf.raw_ops.InitializeTableV2(
        table_handle=table_handle, keys=keys, values=values
    )

    x = tf.placeholder(tf.string, shape=[2], name="x")
    y = tf.placeholder(tf.int32, shape=[], name="y")
    z = tf.raw_ops.LookupTableFindV2(table_handle=table_handle, keys=x, default_value=y)

    rmdir("lookuptablefindv2")
    with tf.Session() as sess:
        sess.run(op)
        tf.saved_model.simple_save(
            sess,
            "lookuptablefindv2",
            inputs={"x": x, "y": y},
            outputs={"z": z},
            legacy_init_op=op,
        )
        print(sess.run([z], feed_dict={"x:0": ["a", "f"], "y:0": -1}))

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], "lookuptablefindv2")
        graph = tf.get_default_graph()
        z = graph.get_tensor_by_name("LookupTableFindV2:0")
        print(sess.run([z], feed_dict={"x:0": ["a", "f"], "y:0": -1}))


@_register_func
def sparsecross():
    tf.reset_default_graph()
    x_indices = tf.constant([[0, 0], [1, 0], [1, 1]], tf.int64)
    x_values = tf.placeholder(tf.string, shape=[3])
    x_dense_shape = tf.constant([2, 2], tf.int64)

    y = tf.SparseTensor(values=["1", "2"], indices=[[0, 0], [1, 0]], dense_shape=[2, 1])
    z = tf.constant([["+", "-"], ["*", "/"]])

    sparse_cross = tf.raw_ops.SparseCross(
        indices=[x_indices, y.indices],
        values=[x_values, y.values],
        shapes=[x_dense_shape, y.dense_shape],
        dense_inputs=[z],
        hashed_output=True,
        num_buckets=100,
        hash_key=956888297470,
        out_type=tf.int64,  # tf.string: get crossed string; tf.int64: get values after hash
        internal_type=tf.int64,
    )
    out_indices = sparse_cross.output_indices
    out_values = sparse_cross.output_values
    out_dense_shape = sparse_cross.output_shape
    out = tf.sparse.to_dense(tf.SparseTensor(out_indices, out_values, out_dense_shape))

    dirname = "sparsecross"
    x_values_numpy = np.array(["a", "b", "c"], dtype="<S1")
    np.savez(dirname + ".npz", x_values=x_values_numpy)
    rmdir(dirname)
    with tf.Session() as sess:
        print(out)
        print(
            sess.run(
                [out],
                feed_dict={"Placeholder:0": x_values_numpy},
            )
        )
        tf.saved_model.simple_save(
            sess,
            dirname,
            inputs={"x_values": x_values},
            outputs={"out": out},
        )


if __name__ == "__main__":
    try:
        args = parser.parse_args()
    except TypeError:
        parser.print_usage()
        exit()
    args.func()
