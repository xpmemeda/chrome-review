import numpy as np
import onnx
from onnx import numpy_helper

a = onnx.helper.make_tensor_value_info("a", onnx.TensorProto.INT64, (2,))
# value_a = onnx.numpy_helper.from_array(np.array([1, 1], np.int64), 'a')

b = onnx.helper.make_tensor_value_info("b", onnx.TensorProto.INT64, (2,))
# value_b = onnx.numpy_helper.from_array(np.array([1, 1], np.int64), 'b')

c = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.INT64, (2,))
value_c = onnx.numpy_helper.from_array(np.array([1, 4], np.int64), "c")

i = onnx.helper.make_tensor_value_info("i", onnx.TensorProto.FLOAT, (1, 4))
o = onnx.helper.make_tensor_value_info("o", onnx.TensorProto.FLOAT, (1, 4))


add_a_b = onnx.helper.make_node("Add", ["a", "b"], ["add_a_b:0"], "add_a_b")
reshape_i1 = onnx.helper.make_node(
    "Reshape", ["i", "add_a_b:0"], ["reshape_i1:0"], "reshape_i1"
)
reshape_i2 = onnx.helper.make_node(
    "Reshape", ["reshape_i1:0", "c"], ["o"], "reshape_i2"
)

graph = onnx.helper.make_graph(
    [add_a_b, reshape_i1, reshape_i2], "graph", [i, a, b, c], [o], initializer=[value_c]
)
model = onnx.helper.make_model(graph)

onnx.checker.check_model(model)
with open("xp.onnx", "wb") as f:
    f.write(model.SerializeToString())


import onnxruntime as ort

onnxnet = ort.InferenceSession("xp.onnx")
o = onnxnet.run(
    None,
    {
        "i": np.array([[3, 2, 1, 0]], np.float32),
        "a": np.array([1, 1], np.int64),
        "b": np.array([1, 1], np.int64),
    },
)[0]
print(o)
