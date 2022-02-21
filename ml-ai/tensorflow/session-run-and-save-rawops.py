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
def bucketize():
    x = tf.placeholder(tf.float32, [2, 2])
    y = tf.raw_ops.Bucketize(input=x, boundaries=[-1, 0, 1])

    dirname = "bucketize"
    x_numpy = np.array([[-4, 2], [0, 1]]).astype(np.float32)
    np.savez(dirname + ".npz", x=x_numpy)

    rmdir(dirname)
    with tf.Session() as sess:
        print(sess.run([y], feed_dict={"Placeholder:0": x_numpy}))
        tf.saved_model.simple_save(sess, dirname, inputs={"x": x}, outputs={"y": y})


@_register_func
def assign():
    x = tf.placeholder(tf.float32, [2, 2], "x")
    y = tf.raw_ops.VariableV2(shape=[2, 2], dtype=tf.float32)
    y = tf.raw_ops.Assign(ref=y, value=x)

    dirname = "assign"
    np.random.seed(0)
    x_numpy = np.random.random(size=[2, 2])
    np.savez(dirname + ".npz", x=x_numpy)

    rmdir(dirname)
    with tf.Session() as sess:
        print(sess.run([y], feed_dict={"x:0": x_numpy}))
        tf.saved_model.simple_save(sess, dirname, inputs={"x": x}, outputs={"y": y})


@_register_func
def switch_merge():
    x = tf.placeholder(tf.float32, [2, 2], "x")
    switch = tf.raw_ops.Switch(data=x, pred=True)
    merge = tf.raw_ops.Merge(inputs=[switch.output_false, switch.output_true])
    y = merge.output

    dirname = "switch_merge"

    np.random.seed(0)
    x_numpy = np.random.random(size=(2, 2))
    np.savez(dirname + ".npz", x=x_numpy)

    rmdir(dirname)
    with tf.Session() as sess:
        print(sess.run([y], feed_dict={"x:0": x_numpy}))
        tf.saved_model.simple_save(sess, dirname, inputs={"x": x}, outputs={"y": y})


@_register_func
def sparsesegmentsum():
    x = tf.placeholder(tf.float32, [3, 4])
    indices = tf.constant([0, 1, 2], tf.int64)
    segment_ids = tf.constant([0, 0, 1], tf.int64)
    y = tf.raw_ops.SparseSegmentSum(data=x, indices=indices, segment_ids=segment_ids)

    dirname = "sparsesegmentsum"
    np.random.seed(0)
    x_numpy = np.random.random(size=(3, 4))
    np.savez(dirname + ".npz", x=x_numpy)

    rmdir(dirname)
    with tf.Session() as sess:
        print(sess.run([y], feed_dict={"Placeholder:0": x_numpy}))
        tf.saved_model.simple_save(sess, dirname, inputs={"x": x}, outputs={"y": y})


@_register_func
def unique():
    x = tf.placeholder(tf.int64, [None], "x")
    r = tf.raw_ops.Unique(x=x)[0]

    dirname = "unique"
    x_numpy = np.array([1, 2, 3, 1, 2, 3, 4, 5, 6, 1], dtype=np.int64)
    np.savez(dirname + ".npz", x=x_numpy)

    rmdir(dirname)
    with tf.Session() as sess:
        print(sess.run([r], feed_dict={"x:0": x_numpy}))
        tf.saved_model.simple_save(sess, dirname, inputs={"x": x}, outputs={"r": r})


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
def stringtohashbucketstrong():
    tf.reset_default_graph()
    x = tf.placeholder(tf.string, [2, 3], "x")
    r = tf.raw_ops.StringToHashBucketStrong(input=x, num_buckets=100, key=[2, 3])

    dirname = "stringtohashbucketstrong"
    rmdir(dirname)
    x_numpy = np.array([["a", "b", "c"], ["d", "e", "f"]], dtype="S1")
    np.savez(dirname + ".npz", x=x_numpy)
    with tf.Session() as sess:
        print(sess.run([r], feed_dict={"x:0": x_numpy}))
        tf.saved_model.simple_save(sess, dirname, inputs={"x": x}, outputs={"r": r})


@_register_func
def sparsefillemptyrows():
    tf.reset_default_graph()
    x = tf.placeholder(tf.int64, [3, 1], "x")
    y = tf.placeholder(tf.float32, [3], "y")
    dense_shape = tf.constant([5], tf.int64)
    default_val = tf.constant(-1, tf.float32)
    r = tf.raw_ops.SparseFillEmptyRows(
        indices=x, values=y, dense_shape=dense_shape, default_value=default_val
    )

    dirname = "sparsefillemptyrows"
    rmdir(dirname)
    x_numpy = np.array([[0], [1], [3]]).astype(np.int64)
    y_numpy = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    np.savez(dirname + ".npz", x=x_numpy, y=y_numpy)
    with tf.Session() as sess:
        print(
            sess.run(
                [r.output_indices, r.output_values, r.empty_row_indicator],
                feed_dict={"x:0": x_numpy, "y:0": y_numpy},
            )
        )
        tf.saved_model.simple_save(
            sess,
            dirname,
            inputs={"x": x, "y": y},
            outputs={"r_indices": r.output_indices, "r_values": r.output_values},
        )


@_register_func
def sparsetodense():
    tf.reset_default_graph()
    x = tf.placeholder(tf.string, [2], "x")
    r = tf.raw_ops.SparseToDense(
        sparse_indices=[[1, 2], [0, 0]],
        output_shape=[2, 3],
        sparse_values=x,
        default_value=".....",
    )

    dirname = "sparsetodense"
    rmdir(dirname)
    x_numpy = np.array(["hello", "world"], dtype="S5")
    np.savez(dirname + ".npz", x=x_numpy)
    with tf.Session() as sess:
        print(sess.run([r], feed_dict={"x:0": x_numpy}))
        tf.saved_model.simple_save(sess, dirname, inputs={"x": x}, outputs={"r": r})


@_register_func
def hashtable():
    tf.reset_default_graph()
    keys = tf.constant(["aa", "ab", "ac", "ad"], tf.string)
    values = tf.constant([1, 2, 3, 4], tf.int64)
    table_handle = tf.raw_ops.HashTableV2(
        key_dtype=tf.string, value_dtype=tf.int64, shared_name="run-raw-ops"
    )
    op = tf.raw_ops.InitializeTableV2(
        table_handle=table_handle, keys=keys, values=values
    )

    x = tf.placeholder(tf.string, shape=[2], name="x")
    y = tf.raw_ops.LookupTableFindV2(
        table_handle=table_handle, keys=x, default_value=tf.constant(-1, tf.int64)
    )
    y = tf.raw_ops.Add(
        x=y, y=tf.raw_ops.LookupTableSizeV2(table_handle=table_handle), name="y"
    )

    dirname = "hashtable"
    x_numpy = np.array(["ab", "af"], dtype="<S2")
    np.savez(dirname + ".npz", x=x_numpy)

    rmdir(dirname)
    with tf.Session() as sess:
        sess.run(op)
        tf.saved_model.simple_save(
            sess,
            dirname,
            inputs={"x": x},
            outputs={"y": y},
            legacy_init_op=op,
        )
        print(sess.run([y], feed_dict={"x:0": x_numpy}))

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], dirname)
        graph = tf.get_default_graph()
        y = graph.get_tensor_by_name("y:0")
        print(sess.run([y], feed_dict={"x:0": x_numpy}))


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


@_register_func
def notequal():
    tf.reset_default_graph()
    x = tf.placeholder(tf.string, [2, 2], "x")
    y = tf.constant([["a", "b"], ["c", "d"]], tf.string)
    r = tf.raw_ops.NotEqual(x=x, y=y, name="r")

    dirname = "notequal"
    x_numpy = np.array(["a", "b", "c", "e"]).astype("S1").reshape([2, 2])
    np.savez(dirname + ".npz", x=x_numpy)
    rmdir(dirname)
    with tf.Session() as sess:
        print(sess.run([r], feed_dict={"x:0": x_numpy}))
        tf.saved_model.simple_save(sess, dirname, inputs={"x": x}, outputs={"r": r})


if __name__ == "__main__":
    try:
        args = parser.parse_args()
    except TypeError:
        parser.print_usage()
        exit()
    args.func()
