import onnx
import numpy as np
from onnx import numpy_helper

node = onnx.helper.make_node(
    'ScatterND',
    inputs=['data', 'indices', 'updates'],
    outputs=['output'],
)

data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, (4, 4, 4))
np_data = np.array(
    [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)

indices = onnx.helper.make_tensor_value_info("indices", onnx.TensorProto.INT64, (2, 1))
np_indices = np.array([[0], [2]], dtype=np.int64)

updates = onnx.helper.make_tensor_value_info("updates", onnx.TensorProto.FLOAT, (2, 4, 4))
np_updates = np.array(
    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=np.float32)

output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, (4, 4, 4))

np_output = np.array(
    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)


graph = onnx.helper.make_graph(
    [node],
    "graph",
    [data, indices, updates],
    [output],
    initializer=[]
)
model = onnx.helper.make_model(graph)

onnx.checker.check_model(model)
with open("scatter_nd.onnx", "wb") as f:
    f.write(model.SerializeToString())

import onnxruntime as ort
onnxnet = ort.InferenceSession("scatter_nd.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
np_output = onnxnet.run(None, {"data": np_data, "indices": np_indices, "updates": np_updates})[0]
print(np_output)
