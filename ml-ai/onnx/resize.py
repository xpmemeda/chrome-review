import onnx
import numpy as np
from onnx import numpy_helper

node = onnx.helper.make_node(
    'Resize',
    inputs=['x', '', '', 'sizes'],
    outputs=['y'],
    mode='nearest',
    coordinate_transformation_mode='half_pixel',
    nearest_mode='round_prefer_ceil',
)


x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 1, 2, 2))
np_x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)

sizes = onnx.helper.make_tensor_value_info("sizes", onnx.TensorProto.INT64, (4,))
np_sizes = np.array([1, 1, 7, 8], dtype=np.int64)

y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (1, 1, 7, 8))

graph = onnx.helper.make_graph(
    [node],
    "graph",
    [x, sizes],
    [y],
    initializer=[]
)
model = onnx.helper.make_model(graph)

onnx.checker.check_model(model)
with open("resize.onnx", "wb") as f:
    f.write(model.SerializeToString())

import onnxruntime as ort
onnxnet = ort.InferenceSession("resize.onnx")
np_y = onnxnet.run(None, {"x": np_x, "sizes": np_sizes})[0]
print(np_y)
print(np_y.tolist())