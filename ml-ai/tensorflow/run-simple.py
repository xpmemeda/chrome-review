import tensorflow as tf


# eager
keys_tensor = tf.constant(["a", "b", "c"])
vals_tensor = tf.constant([7, 8, 9])
init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
table = tf.lookup.StaticHashTable(init, default_value=-1)

input_tensor = tf.constant(["a", "f"])
x = table.lookup(input_tensor).numpy()

print(table)
print(x)


tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session.
sess = tf.compat.v1.Session()

# Evaluate the tensor `c`.
print(sess.run(c))  # prints 30.0
