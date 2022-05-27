import onnx
import numpy as np
from onnx import numpy_helper

node = onnx.helper.make_node(
    'Where',
    inputs=['condition', 'x', 'y'],
    outputs=['z'],
)

condition = onnx.helper.make_tensor_value_info("condition", onnx.TensorProto.BOOL, (2, 1))
np_condition = np.array([[1], [1]], dtype=bool)
x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT64, (2, 2))
np_x = np.array([[1, 2], [3, 4]], dtype=np.int64)
y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.INT64, (2, 2))
np_y = np.array([[9, 8], [7, 6]], dtype=np.int64)

z = onnx.helper.make_tensor_value_info("z", onnx.TensorProto.INT64, (2, 2))

graph = onnx.helper.make_graph(
    [node],
    "graph",
    [condition, x, y],
    [z],
    initializer=[]
)
model = onnx.helper.make_model(graph)

onnx.checker.check_model(model)
with open("where.onnx", "wb") as f:
    f.write(model.SerializeToString())

import onnxruntime as ort
onnxnet = ort.InferenceSession("where.onnx")
np_z = onnxnet.run(None, {"condition": np_condition, "x": np_x, "y": np_y})[0]
print(np_z) # [[1, 8], [3, 4]]
