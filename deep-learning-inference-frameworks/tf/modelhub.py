from typing import Optional
import os
import shutil
import inspect
import pdb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()  # need to disable eager in TF2.x

import numpy as np


class Model:
    def __init__(self, workspace):
        self.workspace = workspace

        tf.reset_default_graph()
        np.random.seed(0)

    def modeldir(self):
        return os.path.join(self.workspace, self.__class__.__name__)


class M1(Model):
    r"""
    ```mermaid
    graph LR;
        x(("input: x")) --> Add
        y(("input: y")) --> Add
        z(("input: z")) --> Mul
        Add-->Mul
        Mul-->r(("output: r"))
    ```
    """

    def __call__(self):
        x = tf.placeholder(tf.float32, [None, None], "x")
        y = tf.placeholder(tf.float32, [None, None], "y")
        z = tf.placeholder(tf.float32, [None, None], "z")
        r = tf.raw_ops.Mul(
            x=tf.raw_ops.Add(x=x, y=y, name="op_add"), y=z, name="op_mul"
        )

        np.random.seed(0)
        x_numpy = np.random.random(size=[2, 3]).astype(np.float32)
        y_numpy = np.random.random(size=[2, 3]).astype(np.float32)
        z_numpy = np.random.random(size=[2, 3]).astype(np.float32)

        with tf.Session() as sess:
            r_numpy, *_ = sess.run(
                [r], feed_dict={"x:0": x_numpy, "y:0": y_numpy, "z:0": z_numpy}
            )
            dirname = self.modeldir()
            tf.saved_model.simple_save(
                sess, dirname, inputs={"x": x, "y": y, "z": z}, outputs={"r": r}
            )
            np.savez(f"{dirname}/1.npz", x=x_numpy, y=y_numpy, z=z_numpy, r=r_numpy)


class M2(Model):
    r"""
    ```mermaid
    graph LR;
        x(("input: x")) --> Join
        y(("input: y")) --> Join
        Join --> r(("output: r"))
    ```
    """

    def __call__(self):
        x = tf.placeholder(tf.string, [3], "x")
        y = tf.placeholder(tf.string, [3], "y")
        r = tf.raw_ops.StringJoin(inputs=[x, y])
        x_numpy = np.array(["a", "b", "c"], dtype="<S9")
        y_numpy = np.array(["x", "y", "z"], dtype="<S9")
        dirname = self.modeldir()
        with tf.Session() as sess:
            r_numpy, *_ = sess.run([r], feed_dict={"x:0": x_numpy, "y:0": y_numpy})
            r_numpy = r_numpy.astype("<S19")
            tf.saved_model.simple_save(
                sess, dirname, inputs={"x": x, "y": y}, outputs={"r": r}
            )
            np.savez(f"{dirname}/1.npz", x=x_numpy, y=y_numpy, r=r_numpy)


if __name__ == "__main__":
    workspace = ".modelhub"
    if os.path.exists(workspace):
        shutil.rmtree(workspace)
    os.mkdir(workspace)

    M2(workspace)()
    M1(workspace)()
