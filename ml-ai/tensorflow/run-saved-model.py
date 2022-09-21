import tensorflow.compat.v1 as tf
import pdb
import sys
import numpy as np

tf.load_op_library("/home/olafxiong/workspace/resources/metis/custom_as_string_op.so")

MODEL_PATH = sys.argv[1]
# IMAGE_PATH = sys.argv[2]

# x = np.load(IMAGE_PATH)

with tf.Session(graph=tf.Graph()) as sess:
    # Restore model from the saved_modle file, that is exported by TensorFlow estimator.
    tf.saved_model.loader.load(sess, ["serve"], MODEL_PATH)

    # Get the output node from the graph.
    graph = tf.get_default_graph()
    outputs = []
    table_handle = graph.get_tensor_by_name("hash_table:0")
    k, v = tf.raw_ops.LookupTableExportV2(
        table_handle=table_handle, Tkeys=tf.int32, Tvalues=tf.int32
    )
    # print(output)  # Tensor("hash_table:0", shape=(), dtype=resource)
    outputs.append(k)
    outputs.append(v)
    output = graph.get_tensor_by_name("hash_table_Lookup/LookupTableFindV2:0")
    outputs.append(output)

    # Tensor("StatefulPartitionedCall:0", shape=(None, 1000), dtype=float32)
    print(outputs)

    # Run forward pass with images.
    # embd = sess.run(output, feed_dict={})
    magic = 10
    embd = sess.run(
        outputs,
        feed_dict={
            "Cast_1:0": np.array([i for i in range(magic)]).astype(np.int32),
            "Const_39:0": -1,
        },
    )

    # nums = []
    # for i in embd[0]:
    #     nums.append(i[0].astype(np.int32))
    # counts = []
    # for i in range(magic):
    #     counts.append(nums.count(i))

    # for key, count, find in zip([i for i in range(magic)], counts, embd[1]):
    #     print("key = %d: count = %d, find = %d" % (key, count, find))
    # print(tf.constant(embd[0]))

    # Print out the result.
    # print(type(embd))  # <class 'numpy.ndarray'>
    # print(embd.shape)
    # print(embd[0])
    # print(type(embd[0]))
    print(embd)
    print(embd[0].shape)
