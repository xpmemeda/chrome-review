import os
import sys
import tensorflow.compat.v1 as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
MODEL_PATH = sys.argv[1]


def main():
    tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x
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

    with tf.Session() as sess:
        sess.run(op)
        tf.saved_model.simple_save(
            sess,
            MODEL_PATH,
            inputs={"x": x, "y": y},
            outputs={"z": z},
            legacy_init_op=op,
        )
        print(sess.run([z], feed_dict={"x:0": ["a", "f"], "y:0": -1}))

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], MODEL_PATH)
        graph = tf.get_default_graph()
        z = graph.get_tensor_by_name("LookupTableFindV2:0")
        print(sess.run([z], feed_dict={"x:0": ["a", "f"], "y:0": -1}))


if __name__ == "__main__":
    main()
