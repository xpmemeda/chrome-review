import onnx
import numpy as np
from onnx import numpy_helper

x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 1, 10, 10))
rois = onnx.helper.make_tensor_value_info("rois", onnx.TensorProto.FLOAT, (3, 4))
batch_indices = onnx.helper.make_tensor_value_info("batch_indices", onnx.TensorProto.INT64, (3,))
y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (3, 1, 5, 5))

np_x = np.array(
    [
        [
            [
                [
                    0.2764,
                    0.7150,
                    0.1958,
                    0.3416,
                    0.4638,
                    0.0259,
                    0.2963,
                    0.6518,
                    0.4856,
                    0.7250,
                ],
                [
                    0.9637,
                    0.0895,
                    0.2919,
                    0.6753,
                    0.0234,
                    0.6132,
                    0.8085,
                    0.5324,
                    0.8992,
                    0.4467,
                ],
                [
                    0.3265,
                    0.8479,
                    0.9698,
                    0.2471,
                    0.9336,
                    0.1878,
                    0.4766,
                    0.4308,
                    0.3400,
                    0.2162,
                ],
                [
                    0.0206,
                    0.1720,
                    0.2155,
                    0.4394,
                    0.0653,
                    0.3406,
                    0.7724,
                    0.3921,
                    0.2541,
                    0.5799,
                ],
                [
                    0.4062,
                    0.2194,
                    0.4473,
                    0.4687,
                    0.7109,
                    0.9327,
                    0.9815,
                    0.6320,
                    0.1728,
                    0.6119,
                ],
                [
                    0.3097,
                    0.1283,
                    0.4984,
                    0.5068,
                    0.4279,
                    0.0173,
                    0.4388,
                    0.0430,
                    0.4671,
                    0.7119,
                ],
                [
                    0.1011,
                    0.8477,
                    0.4726,
                    0.1777,
                    0.9923,
                    0.4042,
                    0.1869,
                    0.7795,
                    0.9946,
                    0.9689,
                ],
                [
                    0.1366,
                    0.3671,
                    0.7011,
                    0.6234,
                    0.9867,
                    0.5585,
                    0.6985,
                    0.5609,
                    0.8788,
                    0.9928,
                ],
                [
                    0.5697,
                    0.8511,
                    0.6711,
                    0.9406,
                    0.8751,
                    0.7496,
                    0.1650,
                    0.1049,
                    0.1559,
                    0.2514,
                ],
                [
                    0.7012,
                    0.4056,
                    0.7879,
                    0.3461,
                    0.0415,
                    0.2998,
                    0.5094,
                    0.3727,
                    0.5482,
                    0.0502,
                ],
            ]
        ]
    ],
    dtype=np.float32,
)
np_rois = np.array([[0, 0, 9, 9], [0, 5, 4, 9], [5, 5, 9, 9]], dtype=np.float32)
np_batch_indices = np.array([0, 0, 0], dtype=np.int64)



node = onnx.helper.make_node(
    "RoiAlign",
    inputs=["x", "rois", "batch_indices"],
    outputs=["y"],
    # coordinate_transformation_mode="half_pixel",
    spatial_scale=1.0,
    output_height=5,
    output_width=5,
    sampling_ratio=2,
    mode="max",
)

graph = onnx.helper.make_graph(
    [node],
    "graph",
    [x, rois, batch_indices],
    [y],
    initializer=[]
)
model = onnx.helper.make_model(graph)

onnx.checker.check_model(model)
with open("roialign.onnx", "wb") as f:
    f.write(model.SerializeToString())

import onnxruntime as ort
onnxnet = ort.InferenceSession("roialign.onnx")
np_y = onnxnet.run(None, {"x": np_x, "rois": np_rois, "batch_indices": np_batch_indices})[0]
print(np_y.tolist())