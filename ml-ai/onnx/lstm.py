import numpy as np
import onnx
from onnx import numpy_helper

seq_length = 3
batch_size = 3
input_size = 2
num_directions = 1
hidden_size = 3
weight_scale = 0.1
number_of_gates = 4

np.random.seed(0)
np_x = np.random.random(size=(seq_length, batch_size, input_size)).astype(np.float32)
np_w = np.random.random(size=(1, number_of_gates * hidden_size, input_size)).astype(np.float32)
np_r = np.random.random(size=(1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
np_seq_len = np.array([3, 2, 1]).astype(np.int32)
np.savez("lstm.npz", x=np_x, w=np_w, r=np_r, seq_len=np_seq_len)

x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (seq_length, batch_size, input_size))
w = onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, (num_directions, number_of_gates * hidden_size, input_size))
r = onnx.helper.make_tensor_value_info("r", onnx.TensorProto.FLOAT, (num_directions, number_of_gates * hidden_size, hidden_size))
seq_len = onnx.helper.make_tensor_value_info("seq_len", onnx.TensorProto.INT32, (batch_size,))
y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (seq_length, num_directions, batch_size, hidden_size))
y_h = onnx.helper.make_tensor_value_info("y_h", onnx.TensorProto.FLOAT, (num_directions, batch_size, hidden_size))
y_c = onnx.helper.make_tensor_value_info("y_c", onnx.TensorProto.FLOAT, (num_directions, batch_size, hidden_size))
node = onnx.helper.make_node(
    'LSTM',
    inputs=['x', 'w', 'r', '', 'seq_len'],
    outputs=['y', 'y_h', 'y_c'],
    hidden_size=hidden_size,
)

graph = onnx.helper.make_graph(
    [node],
    "graph",
    [x, w, r, seq_len],
    [y, y_h, y_c],
    initializer=[]
)
model = onnx.helper.make_model(graph)

onnx.checker.check_model(model)
with open("lstm.onnx", "wb") as f:
    f.write(model.SerializeToString())

import onnxruntime as ort
onnxnet = ort.InferenceSession("lstm.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
np_y, np_y_h, np_y_c = onnxnet.run(None, {"x": np_x, "w": np_w, "r": np_r, "seq_len": np_seq_len})
for name, value in zip(['y', 'y_h', 'y_c'], [np_y, np_y_h, np_y_c]):
    print(
        '[', name, ']',
        ' sum: ', np.sum(value),
        ' avg: ', np.mean(value),
        ' var: ', np.var(value),
        ' max: ', np.max(value),
        ' min: ', np.min(value)
    )
    print(value.shape)
    print(value)
