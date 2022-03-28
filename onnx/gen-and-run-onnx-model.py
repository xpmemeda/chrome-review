import numpy as np
import onnx
from onnx import numpy_helper

i = onnx.helper.make_tensor_value_info("i", onnx.TensorProto.FLOAT, (2, 2))
o = onnx.helper.make_tensor_value_info("o", onnx.TensorProto.FLOAT, (2, 2))

# All parameters should be graph's inputs, but with value
ct = onnx.helper.make_tensor_value_info("c", onnx.TensorProto.FLOAT, (2, 2))
cp = onnx.numpy_helper.from_array(np.array([[0, 1], [2, 3]], np.float32), "c")

identity = onnx.helper.make_node("Identity", ["i"], ["identity"], "identity")
add = onnx.helper.make_node("Add", ["identity", "c"], ["o"], "add")
graph = onnx.helper.make_graph([identity, add], "graph", [i, ct], [o], initializer=[cp])
model = onnx.helper.make_model(graph)

onnx.checker.check_model(model)
with open("xp.onnx", "wb") as f:
    f.write(model.SerializeToString())


import onnxruntime as ort
onnxnet = ort.InferenceSession("xp.onnx")
o = onnxnet.run(None, {"i": np.array([[3, 2], [1, 0]], np.float32)})[0]
print(o)
