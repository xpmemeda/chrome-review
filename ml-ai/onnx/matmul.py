import numpy as np
import onnx
from onnx import numpy_helper

np.random.seed(0)
np_x = np.random.random(size=(2, 3, 4)).astype(np.float32)
np_w = np.random.random(size=(2, 4, 5)).astype(np.float32)

# in
x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (2, 3, 4))
# perameter
w = onnx.helper.make_tensor(
    "w", onnx.TensorProto.FLOAT, [2, 4, 5], np_w.reshape(-1).tolist()
)
# out
y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 3, 5))

node = onnx.helper.make_node("MatMul", inputs=["x", "w"], outputs=["y"])
graph = onnx.helper.make_graph([node], "graph", [x], [y], initializer=[w])
model = onnx.helper.make_model(graph)

onnx.checker.check_model(model)
with open("matmul.onnx", "wb") as f:
    f.write(model.SerializeToString())

import onnxruntime as ort

onnxnet = ort.InferenceSession("matmul.onnx", providers=["CPUExecutionProvider"])
np_y = onnxnet.run(None, {"x": np_x})
for name, value in zip(["y"], np_y):
    print(
        "[",
        name,
        "]",
        " sum: ",
        np.sum(value),
        " avg: ",
        np.mean(value),
        " var: ",
        np.var(value),
        " max: ",
        np.max(value),
        " min: ",
        np.min(value),
    )
    print(value.shape)
