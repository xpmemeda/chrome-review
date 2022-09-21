import tensorflow as tf

keys_tensor = tf.constant(["a", "b", "c"])
vals_tensor = tf.constant([7, 8, 9])
input_tensor = tf.constant(["a", "f"])
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=-1
)
x = table.lookup(input_tensor).numpy()

print(x)


class CustomModule(tf.Module):
    def __init__(self):
        super(CustomModule, self).__init__()
        self.v = tf.Variable(1.0)

    @tf.function
    def __call__(self, x):
        """
        Function Name: '__call__'
            ...some...description...

        signature(if specified):

        inputs['x'] tensor_info:
            dtype: DT_FLOAT
            shape: unknown_rank
            name: serving_default_x:0
        """
        print("Tracing with", x)
        return x * self.v

    @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
    def mutate(self, new_v):
        """
        Function Name: 'mutate'
            ...some...description...

        signature(if specified):

        signature:

        inputs['new_v'] tensor_info:
            dtype: DT_FLOAT
            shape: ()
            name: array_input_new_v:0
        """
        self.v.assign(new_v)
        return tf.constant(["a", "b", "c"])


module = CustomModule()
import os

# saved_model_cli show --all --dir module_with_multiple_signatures
# (print model info)

module_with_signature_path = os.path.join("module_with_signature")
call = module.__call__.get_concrete_function(tf.TensorSpec(None, tf.float32))
tf.saved_model.save(module, module_with_signature_path, signatures=call)

module_multiple_signatures_path = os.path.join("module_with_multiple_signatures")
signatures = {
    "serving_default": call,
    "array_input": module.mutate.get_concrete_function(tf.TensorSpec([], tf.float32)),
}
tf.saved_model.save(module, module_multiple_signatures_path, signatures=signatures)

module_no_signatures_path = os.path.join("module_no_signatures")
module(tf.constant(0.0))
print("Saving model...")
tf.saved_model.save(module, module_no_signatures_path)


imported = tf.saved_model.load(module_no_signatures_path)
print(type(imported))
assert imported(tf.constant(3.0)).numpy() == 3
imported.mutate(tf.constant(2.0))
assert imported(tf.constant(3.0)).numpy() == 6

import tensorflow.compat.v1 as tf1

with tf1.Session(graph=tf1.Graph()) as sess:
    # Restore model from the saved_modle file, that is exported by TensorFlow estimator.
    tf1.saved_model.loader.load(sess, ["serve"], "module_with_multiple_signatures")

    # Get the output node from the graph.
    graph = tf1.get_default_graph()
    output = graph.get_tensor_by_name("StatefulPartitionedCall:0")

    # Tensor("StatefulPartitionedCall:0", shape=(None, 1000), dtype=float32)
    print(output)

    # Run forward pass with images.
    # The value of a feed cannot be a tf.Tensor object. Acceptable feed values include Python scalars, strings, lists, numpy ndarrays, or TensorHandles.
    embd = sess.run(output, feed_dict={"array_input_new_v:0": 2.0})

    # Print out the result.
    print(type(embd))  # <class 'numpy.ndarray'>
    print(embd)
