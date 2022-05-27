import numpy as np
import onnx
from onnx import numpy_helper

np.random.seed(0)
np_x = np.random.random(size=(1, 1, 3, 3, 3)).astype(np.float32)
np_w = np.random.random(size=(1, 1, 3, 3, 3)).astype(np.float32)

x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 1, 3, 3, 3))
w = onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, (1, 1, 3, 3, 3))
y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (1, 1, 3, 3, 3))

# Convolution with padding
node = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'w'],
    outputs=['y'],
    kernel_shape=[3, 3, 3],
    pads=[1, 1, 1, 1, 1, 1],
)

graph = onnx.helper.make_graph(
    [node],
    "graph",
    [x, w],
    [y],
    initializer=[]
)
model = onnx.helper.make_model(graph)

onnx.checker.check_model(model)
with open("conv.onnx", "wb") as f:
    f.write(model.SerializeToString())

import onnxruntime as ort
onnxnet = ort.InferenceSession("conv.onnx", providers=['CPUExecutionProvider'])
np_y = onnxnet.run(None, {"x": np_x, "w": np_w})
for name, value in zip(['y'], [np_y]):
    print(
        '[', name, ']',
        ' sum: ', np.sum(value),
        ' avg: ', np.mean(value),
        ' var: ', np.var(value),
        ' max: ', np.max(value),
        ' min: ', np.min(value)
    )
    # print(value.shape)
    print(value)
