import onnx
import numpy as np
from onnx import numpy_helper

boxes = onnx.helper.make_tensor_value_info("boxes", onnx.TensorProto.FLOAT, (2, 6, 4,))
scores = onnx.helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, (2, 2, 6,))

max_output_boxes_per_class = onnx.helper.make_tensor_value_info("max_output_boxes_per_class", onnx.TensorProto.INT64, (1,))
np_max_output_boxes_per_class = np.array([3]).astype(np.int64)
value_max_output_boxes_per_class = onnx.numpy_helper.from_array(np_max_output_boxes_per_class, 'max_output_boxes_per_class')

iou_threshold = onnx.helper.make_tensor_value_info("iou_threshold", onnx.TensorProto.FLOAT, (1,))
np_iou_threshold = np.array([0.5]).astype(np.float32)
value_iou_threshold = onnx.numpy_helper.from_array(np_iou_threshold, 'iou_threshold')

score_threshold = onnx.helper.make_tensor_value_info("score_threshold", onnx.TensorProto.FLOAT, (1,))
np_score_threshold = np.array([0.0]).astype(np.float32)
value_score_threshold = onnx.numpy_helper.from_array(np_score_threshold, 'score_threshold')

selected_indices = onnx.helper.make_tensor_value_info("selected_indices", onnx.TensorProto.INT64, (12, 3))

node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices'],
    center_point_box=1
)

np_boxes = np.array([[
    [0.5, 0.5, 1.0, 1.0],
    [0.5, 0.6, 1.0, 1.0],
    [0.5, 0.4, 1.0, 1.0],
    [0.5, 10.5, 1.0, 1.0],
    [0.5, 10.6, 1.0, 1.0],
    [0.5, 100.5, 1.0, 1.0]
], [
    [0.5, 0.5, 1.0, 1.0],
    [0.5, 0.6, 1.0, 1.0],
    [0.5, 0.4, 1.0, 1.0],
    [0.5, 10.5, 1.0, 1.0],
    [0.5, 10.6, 1.0, 1.0],
    [0.5, 100.5, 1.0, 1.0]]]).astype(np.float32)
np_scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3], [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]], [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3], [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
np_selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

graph = onnx.helper.make_graph(
    [node],
    "graph",
    [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
    [selected_indices],
    initializer=[value_max_output_boxes_per_class, value_iou_threshold, value_score_threshold]
)
model = onnx.helper.make_model(graph)

onnx.checker.check_model(model)
with open("xp.onnx", "wb") as f:
    f.write(model.SerializeToString())

import onnxruntime as ort
onnxnet = ort.InferenceSession("xp.onnx")
o = onnxnet.run(None, {"boxes": np_boxes, "scores": np_scores})[0]
print(o)
