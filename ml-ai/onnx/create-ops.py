import numpy as np
import onnx
import onnxruntime as ort
import argparse


parser = argparse.ArgumentParser(add_help=True)
subparsers = parser.add_subparsers(help="cmd")
subparsers.required = True


def print_numpy_array(arr):
    print(arr.reshape([-1]).tolist()[:3])
    print(arr.shape)
    print(
        "min: {:.2f}, max: {:.2f}, avg: {:.2f}, var: {:.2f}".format(
            np.min(arr), np.max(arr), np.mean(arr), np.var(arr)
        )
    )


def _register_func(impl):
    subparser = subparsers.add_parser(impl.__name__)
    subparser.set_defaults(func=impl)


@_register_func
def slice():
    slice = onnx.helper.make_node("Slice", inputs=['x', 'starts', 'ends', 'axes', 'steps'], outputs=['r'])

    x = onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [3, 4])
    x_numpy = np.random.random(size=[3, 4]).astype(np.float32)
    print(x_numpy)
    starts = onnx.helper.make_tensor_value_info('starts', onnx.TensorProto.INT64, [1])
    starts_numpy = np.array([-1]).astype(np.int64)
    ends = onnx.helper.make_tensor_value_info('ends', onnx.TensorProto.INT64, [1])
    ends_numpy = np.array([-200]).astype(np.int64)
    axes = onnx.helper.make_tensor_value_info('axes', onnx.TensorProto.INT64, [1])
    axes_numpy = np.array([0]).astype(np.int64)
    steps = onnx.helper.make_tensor_value_info('steps', onnx.TensorProto.INT64, [1])
    steps_numpy = np.array([-1]).astype(np.int64)

    r = onnx.helper.make_tensor_value_info('r', onnx.TensorProto.FLOAT, [3, 4])

    g = onnx.helper.make_graph([slice], 'g', [x, starts, ends, axes, steps], [r], initializer=[])
    m = onnx.helper.make_model(g)
    onnx.checker.check_model(m)
    with open('slice.onnx', 'wb') as f:
        f.write(m.SerializeToString())

    onnxnet = ort.InferenceSession('slice.onnx', providers=['CPUExecutionProvider'])
    r_numpy = onnxnet.run(None, {'x': x_numpy, 'starts': starts_numpy, 'ends': ends_numpy, 'axes': axes_numpy, 'steps': steps_numpy})[0]
    print(r_numpy)


@_register_func
def where():
    node = onnx.helper.make_node(
        "Where",
        inputs=["condition", "x", "y"],
        outputs=["z"],
    )

    condition = onnx.helper.make_tensor_value_info(
        "condition", onnx.TensorProto.BOOL, (2, 1)
    )
    np_condition = np.array([[1], [1]], dtype=bool)
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT64, (2, 2))
    np_x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.INT64, (2, 2))
    np_y = np.array([[9, 8], [7, 6]], dtype=np.int64)

    z = onnx.helper.make_tensor_value_info("z", onnx.TensorProto.INT64, (2, 2))

    graph = onnx.helper.make_graph(
        [node], "graph", [condition, x, y], [z], initializer=[]
    )
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("where.onnx", "wb") as f:
        f.write(model.SerializeToString())

    onnxnet = ort.InferenceSession("where.onnx")
    np_z = onnxnet.run(None, {"condition": np_condition, "x": np_x, "y": np_y})[0]
    return np_z


@_register_func
def resize():
    node = onnx.helper.make_node(
        "Resize",
        inputs=["x", "", "", "sizes"],
        outputs=["y"],
        mode="nearest",
        coordinate_transformation_mode="half_pixel",
        nearest_mode="round_prefer_ceil",
    )

    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 1, 2, 2))
    np_x = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)

    sizes = onnx.helper.make_tensor_value_info("sizes", onnx.TensorProto.INT64, (4,))
    np_sizes = np.array([1, 1, 7, 8], dtype=np.int64)

    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, (1, 1, 7, 8))

    graph = onnx.helper.make_graph([node], "graph", [x, sizes], [y], initializer=[])
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("resize.onnx", "wb") as f:
        f.write(model.SerializeToString())

    onnxnet = ort.InferenceSession("resize.onnx")
    np_y = onnxnet.run(None, {"x": np_x, "sizes": np_sizes})[0]
    return np_y


@_register_func
def expand():
    x_shp = [3, 1]
    y_shp = [3]
    r_shp = [2, 3, 4]
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, x_shp)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.INT64, y_shp)
    r = onnx.helper.make_tensor_value_info("r", onnx.TensorProto.FLOAT, r_shp)

    expand = onnx.helper.make_node("Expand", inputs=["x", "y"], outputs=["r"])
    g = onnx.helper.make_graph([expand], "g", [x, y], [r])
    m = onnx.helper.make_model(g)

    onnx.checker.check_model(m)
    with open("expand.onnx", "wb") as f:
        f.write(m.SerializeToString())

    np.random.seed(0)
    x_numpy = np.random.randint(-128, 128, size=[3, 1]).astype(np.float32)
    y_numpy = np.array([2, 3, 4]).astype(np.int64)
    ins = {"x": x_numpy, "y": y_numpy}
    np.savez("expand.npz", **ins)

    onnxnet = ort.InferenceSession("expand.onnx", providers=["CPUExecutionProvider"])
    r_numpy = onnxnet.run(["r"], ins)
    print(r_numpy)


@_register_func
def not_():
    x_shp = [2, 3]
    y_shp = [2, 3]
    r_shp = [2, 3]
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.INT64, x_shp)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.INT64, y_shp)
    r = onnx.helper.make_tensor_value_info("r", onnx.TensorProto.BOOL, r_shp)

    equal = onnx.helper.make_node("Equal", inputs=["x", "y"], outputs=["equal"])
    not_ = onnx.helper.make_node("Not", inputs=["equal"], outputs=["r"])
    g = onnx.helper.make_graph([equal, not_], "g", [x, y], [r])
    m = onnx.helper.make_model(g)

    onnx.checker.check_model(m)
    with open("not.onnx", "wb") as f:
        f.write(m.SerializeToString())

    x_numpy = np.array([0, 1, 0, 2, 0, 3]).reshape(x_shp).astype(np.int64)
    y_numpy = np.array([0, 1, 2, 2, 3, 0]).reshape(y_shp).astype(np.int64)
    ins = {"x": x_numpy, "y": y_numpy}
    np.savez("not.npz", **ins)

    onnxnet = ort.InferenceSession("not.onnx", providers=["CPUExecutionProvider"])
    r_numpy = onnxnet.run(["r"], ins)
    print(r_numpy)


@_register_func
def scatter_nd():
    node = onnx.helper.make_node(
        "ScatterND",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
    )

    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, (4, 4, 4))
    np_data = np.array(
        [
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
        ],
        dtype=np.float32,
    )

    indices = onnx.helper.make_tensor_value_info(
        "indices", onnx.TensorProto.INT64, (2, 1)
    )
    np_indices = np.array([[0], [2]], dtype=np.int64)

    updates = onnx.helper.make_tensor_value_info(
        "updates", onnx.TensorProto.FLOAT, (2, 4, 4)
    )
    np_updates = np.array(
        [
            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ],
        dtype=np.float32,
    )

    output = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, (4, 4, 4)
    )

    np_output = np.array(
        [
            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
        ],
        dtype=np.float32,
    )

    graph = onnx.helper.make_graph(
        [node], "graph", [data, indices, updates], [output], initializer=[]
    )
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("scatter_nd.onnx", "wb") as f:
        f.write(model.SerializeToString())

    onnxnet = ort.InferenceSession(
        "scatter_nd.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    np_output = onnxnet.run(
        None, {"data": np_data, "indices": np_indices, "updates": np_updates}
    )[0]
    return np_output


@_register_func
def nonmaxsupperssion():
    boxes = onnx.helper.make_tensor_value_info(
        "boxes",
        onnx.TensorProto.FLOAT,
        (
            2,
            6,
            4,
        ),
    )
    scores = onnx.helper.make_tensor_value_info(
        "scores",
        onnx.TensorProto.FLOAT,
        (
            2,
            2,
            6,
        ),
    )

    max_output_boxes_per_class = onnx.helper.make_tensor_value_info(
        "max_output_boxes_per_class", onnx.TensorProto.INT64, (1,)
    )
    np_max_output_boxes_per_class = np.array([3]).astype(np.int64)
    value_max_output_boxes_per_class = onnx.numpy_helper.from_array(
        np_max_output_boxes_per_class, "max_output_boxes_per_class"
    )

    iou_threshold = onnx.helper.make_tensor_value_info(
        "iou_threshold", onnx.TensorProto.FLOAT, (1,)
    )
    np_iou_threshold = np.array([0.5]).astype(np.float32)
    value_iou_threshold = onnx.numpy_helper.from_array(
        np_iou_threshold, "iou_threshold"
    )

    score_threshold = onnx.helper.make_tensor_value_info(
        "score_threshold", onnx.TensorProto.FLOAT, (1,)
    )
    np_score_threshold = np.array([0.0]).astype(np.float32)
    value_score_threshold = onnx.numpy_helper.from_array(
        np_score_threshold, "score_threshold"
    )

    selected_indices = onnx.helper.make_tensor_value_info(
        "selected_indices", onnx.TensorProto.INT64, (12, 3)
    )

    node = onnx.helper.make_node(
        "NonMaxSuppression",
        inputs=[
            "boxes",
            "scores",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        outputs=["selected_indices"],
        center_point_box=1,
    )

    np_boxes = np.array(
        [
            [
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.6, 1.0, 1.0],
                [0.5, 0.4, 1.0, 1.0],
                [0.5, 10.5, 1.0, 1.0],
                [0.5, 10.6, 1.0, 1.0],
                [0.5, 100.5, 1.0, 1.0],
            ],
            [
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.6, 1.0, 1.0],
                [0.5, 0.4, 1.0, 1.0],
                [0.5, 10.5, 1.0, 1.0],
                [0.5, 10.6, 1.0, 1.0],
                [0.5, 100.5, 1.0, 1.0],
            ],
        ]
    ).astype(np.float32)
    np_scores = np.array(
        [
            [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3], [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
            [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3], [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
        ]
    ).astype(np.float32)
    np_selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

    graph = onnx.helper.make_graph(
        [node],
        "graph",
        [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
        [selected_indices],
        initializer=[
            value_max_output_boxes_per_class,
            value_iou_threshold,
            value_score_threshold,
        ],
    )
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("xp.onnx", "wb") as f:
        f.write(model.SerializeToString())

    onnxnet = ort.InferenceSession("xp.onnx")
    o = onnxnet.run(None, {"boxes": np_boxes, "scores": np_scores})[0]
    return o


@_register_func
def roialign():
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 1, 10, 10))
    rois = onnx.helper.make_tensor_value_info("rois", onnx.TensorProto.FLOAT, (3, 4))
    batch_indices = onnx.helper.make_tensor_value_info(
        "batch_indices", onnx.TensorProto.INT64, (3,)
    )
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
        [node], "graph", [x, rois, batch_indices], [y], initializer=[]
    )
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("roialign.onnx", "wb") as f:
        f.write(model.SerializeToString())

    onnxnet = ort.InferenceSession("roialign.onnx")
    np_y = onnxnet.run(
        None, {"x": np_x, "rois": np_rois, "batch_indices": np_batch_indices}
    )[0]
    return np_y


@_register_func
def lstm():
    seq_length = 3
    batch_size = 3
    input_size = 2
    num_directions = 1
    hidden_size = 3
    weight_scale = 0.1
    number_of_gates = 4

    np.random.seed(0)
    np_x = np.random.random(size=(seq_length, batch_size, input_size)).astype(
        np.float32
    )
    np_w = np.random.random(size=(1, number_of_gates * hidden_size, input_size)).astype(
        np.float32
    )
    np_r = np.random.random(
        size=(1, number_of_gates * hidden_size, hidden_size)
    ).astype(np.float32)
    np_seq_len = np.array([3, 2, 1]).astype(np.int32)
    np.savez("lstm.npz", x=np_x, w=np_w, r=np_r, seq_len=np_seq_len)

    x = onnx.helper.make_tensor_value_info(
        "x", onnx.TensorProto.FLOAT, (seq_length, batch_size, input_size)
    )
    w = onnx.helper.make_tensor_value_info(
        "w",
        onnx.TensorProto.FLOAT,
        (num_directions, number_of_gates * hidden_size, input_size),
    )
    r = onnx.helper.make_tensor_value_info(
        "r",
        onnx.TensorProto.FLOAT,
        (num_directions, number_of_gates * hidden_size, hidden_size),
    )
    seq_len = onnx.helper.make_tensor_value_info(
        "seq_len", onnx.TensorProto.INT32, (batch_size,)
    )
    y = onnx.helper.make_tensor_value_info(
        "y",
        onnx.TensorProto.FLOAT,
        (seq_length, num_directions, batch_size, hidden_size),
    )
    y_h = onnx.helper.make_tensor_value_info(
        "y_h", onnx.TensorProto.FLOAT, (num_directions, batch_size, hidden_size)
    )
    y_c = onnx.helper.make_tensor_value_info(
        "y_c", onnx.TensorProto.FLOAT, (num_directions, batch_size, hidden_size)
    )
    node = onnx.helper.make_node(
        "LSTM",
        inputs=["x", "w", "r", "", "seq_len"],
        outputs=["y", "y_h", "y_c"],
        hidden_size=hidden_size,
    )

    graph = onnx.helper.make_graph(
        [node], "graph", [x, w, r, seq_len], [y, y_h, y_c], initializer=[]
    )
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("lstm.onnx", "wb") as f:
        f.write(model.SerializeToString())

    onnxnet = ort.InferenceSession(
        "lstm.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    np_y, np_y_h, np_y_c = onnxnet.run(
        None, {"x": np_x, "w": np_w, "r": np_r, "seq_len": np_seq_len}
    )
    return np_y, np_y_h, np_y_c


@_register_func
def matmul3d():
    # 1.7   : thread-num=1, 20ms
    # 2.7.1 : thread-num=1, 20ms
    np.random.seed(0)

    x_shp = [2, 700, 800]
    w_shp = [2, 800, 900]
    r_shp = [2, 700, 900]

    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, x_shp)
    w = onnx.helper.make_tensor(
        "w",
        onnx.TensorProto.FLOAT,
        w_shp,
        np.random.random(size=w_shp).astype(np.float32).reshape([-1]).tolist(),
    )
    r = onnx.helper.make_tensor_value_info("r", onnx.TensorProto.FLOAT, r_shp)

    node = onnx.helper.make_node("MatMul", inputs=["x", "w"], outputs=["r"])
    graph = onnx.helper.make_graph([node], "graph", [x], [r], initializer=[w])
    model = onnx.helper.make_model(graph)

    filename = "matmul3d"
    onnx.checker.check_model(model)
    with open(filename + ".onnx", "wb") as f:
        f.write(model.SerializeToString())

    x_numpy = np.random.random(size=x_shp).astype(np.float32)
    ins = {"x": x_numpy}
    np.savez(filename + ".npz", **ins)

    onnxnet = ort.InferenceSession(
        filename + ".onnx", providers=["CPUExecutionProvider"]
    )
    r_numpy = onnxnet.run(None, ins)
    print_numpy_array(r_numpy[0])


@_register_func
def conv1d():
    x_shp = [2, 3, 5]
    w_shp = [3, 3, 3]
    r_shp = [2, 3, 5]
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, x_shp)
    w = onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, w_shp)
    r = onnx.helper.make_tensor_value_info("r", onnx.TensorProto.FLOAT, r_shp)
    node = onnx.helper.make_node(
        "Conv",
        inputs=["x", "w"],
        outputs=["r"],
        kernel_shape=[3],
        pads=[1, 1],
        dilations=[1],
        strides=[1],
    )
    graph = onnx.helper.make_graph([node], "graph", [x, w], [r], initializer=[])
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("conv1d.onnx", "wb") as f:
        f.write(model.SerializeToString())

    np.random.seed(0)
    x_numpy = np.random.random(size=x_shp).astype(np.float32)
    w_numpy = np.random.random(size=w_shp).astype(np.float32)
    ins = {"x": x_numpy, "w": w_numpy}
    np.savez("conv1d.npz", **ins)
    onnxnet = ort.InferenceSession("conv1d.onnx", providers=["CPUExecutionProvider"])
    print_numpy_array(onnxnet.run(None, ins)[0])


@_register_func
def conv2d():
    x_shp = [2, 8, 8, 8]
    w_shp = [8, 8, 3, 3]
    r_shp = [2, 8, 8, 8]
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, x_shp)
    w = onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, w_shp)
    r = onnx.helper.make_tensor_value_info("r", onnx.TensorProto.FLOAT, r_shp)
    node = onnx.helper.make_node(
        "Conv",
        inputs=["x", "w"],
        outputs=["r"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        strides=[1, 1],
    )
    graph = onnx.helper.make_graph([node], "graph", [x, w], [r], initializer=[])
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("conv2d.onnx", "wb") as f:
        f.write(model.SerializeToString())

    np.random.seed(0)
    x_numpy = np.random.random(size=x_shp).astype(np.float32)
    w_numpy = np.random.random(size=w_shp).astype(np.float32)
    ins = {"x": x_numpy, "w": w_numpy}
    np.savez("conv2d.npz", **ins)
    onnxnet = ort.InferenceSession("conv2d.onnx", providers=["CPUExecutionProvider"])
    print_numpy_array(onnxnet.run(None, ins)[0])


@_register_func
def groupconv2d():
    x_shp = [2, 8, 8, 8]
    w_shp = [8, 2, 3, 3]
    r_shp = [2, 8, 8, 8]
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, x_shp)
    w = onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, w_shp)
    r = onnx.helper.make_tensor_value_info("r", onnx.TensorProto.FLOAT, r_shp)
    node = onnx.helper.make_node(
        "Conv",
        inputs=["x", "w"],
        outputs=["r"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        dilations=[1, 1],
        strides=[1, 1],
        group=4,
    )
    graph = onnx.helper.make_graph([node], "graph", [x, w], [r], initializer=[])
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("groupconv2d.onnx", "wb") as f:
        f.write(model.SerializeToString())

    x_numpy = np.random.random(size=x_shp).astype(np.float32)
    w_numpy = np.random.random(size=w_shp).astype(np.float32)
    ins = {"x": x_numpy, "w": w_numpy}
    np.savez("groupconv2d.npz", **ins)
    onnxnet = ort.InferenceSession(
        "groupconv2d.onnx", providers=["CPUExecutionProvider"]
    )
    print_numpy_array(onnxnet.run(None, ins)[0])


@_register_func
def conv3d():
    x_shp = [2, 8, 8, 8, 8]
    w_shp = [8, 8, 3, 3, 3]
    r_shp = [2, 8, 8, 8, 8]
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, x_shp)
    w = onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, w_shp)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, r_shp)
    node = onnx.helper.make_node(
        "Conv",
        inputs=["x", "w"],
        outputs=["y"],
        kernel_shape=[3, 3, 3],
        pads=[1, 1, 1, 1, 1, 1],
        dilations=[1, 1, 1],
        strides=[1, 1, 1],
    )
    graph = onnx.helper.make_graph([node], "graph", [x, w], [y], initializer=[])
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("conv3d.onnx", "wb") as f:
        f.write(model.SerializeToString())

    np.random.seed(0)
    x_numpy = np.random.random(size=x_shp).astype(np.float32)
    w_numpy = np.random.random(size=w_shp).astype(np.float32)
    ins = {"x": x_numpy, "w": w_numpy}
    np.savez("conv3d.npz", **ins)
    onnxnet = ort.InferenceSession("conv3d.onnx", providers=["CPUExecutionProvider"])
    print_numpy_array(onnxnet.run(None, ins)[0])


@_register_func
def groupconv3d():
    x_shp = [2, 24, 8, 112, 112]
    w_shp = [24, 1, 5, 1, 1]
    y_shp = [2, 24, 8, 112, 112]
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, x_shp)
    w = onnx.helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, w_shp)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, y_shp)
    node = onnx.helper.make_node(
        "Conv",
        inputs=["x", "w"],
        outputs=["y"],
        kernel_shape=[5, 1, 1],
        pads=[2, 0, 0, 2, 0, 0],
        dilations=[1, 1, 1],
        strides=[1, 1, 1],
        group=24,
    )
    graph = onnx.helper.make_graph([node], "graph", [x, w], [y], initializer=[])
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("groupconv3d.onnx", "wb") as f:
        f.write(model.SerializeToString())

    np.random.seed(0)
    x_numpy = np.random.random(size=x_shp).astype(np.float32)
    w_numpy = np.random.random(size=w_shp).astype(np.float32)
    ins = {"x": x_numpy, "w": w_numpy}
    np.savez("groupconv3d.npz", **ins)
    onnxnet = ort.InferenceSession(
        "groupconv3d.onnx", providers=["CPUExecutionProvider"]
    )
    print_numpy_array(onnxnet.run(None, ins)[0])


@_register_func
def einsum():
    x_shp = [2, 2]
    y_shp = [1, 2, 3]
    z_shp = [2, 3]
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, x_shp)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, y_shp)
    z = onnx.helper.make_tensor_value_info("z", onnx.TensorProto.FLOAT, z_shp)

    eqn = "bn,bni->bi"
    node = onnx.helper.make_node(
        "Einsum", inputs=["x", "y"], outputs=["z"], equation=eqn
    )

    graph = onnx.helper.make_graph([node], "graph", [x, y], [z])
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("einsum.onnx", "wb") as f:
        f.write(model.SerializeToString())

    np.random.seed(0)
    np_x = np.random.random(size=x_shp).astype(np.float32)
    np_y = np.random.random(size=y_shp).astype(np.float32)
    np.savez("einsum.npz", x=np_x, y=np_y)
    onnxnet = ort.InferenceSession("einsum.onnx", providers=["CPUExecutionProvider"])
    np_z = onnxnet.run(None, {"x": np_x, "y": np_y})
    print("sum: ", np.sum(np_z))
    print("avg: ", np.mean(np_z))
    print("var: ", np.var(np_z))
    print("max: ", np.max(np_z))
    print("min: ", np.min(np_z))


@_register_func
def gathernd():
    x_shp = [2, 2, 2]
    y_shp = [2, 1, 2]
    r_shp = [2, 1, 2]
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.STRING, x_shp)
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.INT64, y_shp)
    r = onnx.helper.make_tensor_value_info("r", onnx.TensorProto.STRING, r_shp)

    node = onnx.helper.make_node("GatherND", inputs=["x", "y"], outputs=["r"])
    graph = onnx.helper.make_graph([node], "graph", [x, y], [r])
    model = onnx.helper.make_model(graph)

    onnx.checker.check_model(model)
    with open("gathernd.onnx", "wb") as f:
        f.write(model.SerializeToString())

    x_numpy = (
        np.array(["a", "b", "c", "d", "e", "f", "g", "h"])
        .astype("S1")
        .reshape([2, 2, 2])
    )
    y_numpy = np.array([0, 1, 1, 0], dtype=np.int64).reshape([2, 1, 2])
    np.savez("gathernd.npz", x=x_numpy, y=y_numpy)

    onnxnet = ort.InferenceSession("gathernd.onnx", providers=["CPUExecutionProvider"])
    r_numpy = onnxnet.run(["r"], {"x": x_numpy, "y": y_numpy})
    print(r_numpy)


if __name__ == "__main__":
    try:
        args = parser.parse_args()
    except TypeError:
        parser.print_usage()
        exit()
    args.func()
