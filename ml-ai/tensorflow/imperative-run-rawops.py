from numpy import indices
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(add_help=True)
subparsers = parser.add_subparsers(help="cmd")
subparsers.required = True


def _register_func(impl):
    subparser = subparsers.add_parser(impl.__name__)
    subparser.set_defaults(func=impl)


@_register_func
def sparsecross():
    x = tf.SparseTensor(
        values=["a", "b", "c"], indices=[[0, 0], [1, 0], [1, 1]], dense_shape=[2, 2]
    )
    y = tf.SparseTensor(values=["1", "2"], indices=[[0, 0], [1, 0]], dense_shape=[2, 1])
    z = tf.constant([["+", "-"], ["*", "/"]], tf.string)
    print(
        tf.raw_ops.SparseCross(
            values=[x.values, y.values],
            indices=[x.indices, y.indices],
            shapes=[x.dense_shape, y.dense_shape],
            dense_inputs=[z],
            hashed_output=True,
            num_buckets=100,
            hash_key=956888297470,
            out_type=tf.string,
            internal_type=tf.string,
        )
    )


@_register_func
def sparsetodense():
    x = tf.SparseTensor(
        values=["a", "b", "c"], indices=[[0, 0], [1, 0], [1, 1]], dense_shape=[2, 2]
    )
    print(tf.sparse.to_dense(x))


if __name__ == "__main__":
    try:
        args = parser.parse_args()
    except TypeError:
        parser.print_usage()
        exit()
    args.func()
