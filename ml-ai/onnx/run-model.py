import numpy as np
import onnxruntime as ort
import argparse
import onnx
import io


def print_numpy_array(arr):
    arr = arr.reshape([-1]).tolist()[:6]
    arr_str_format = len(arr) * "{:.2f}, "
    arr_str = arr_str_format.format(*arr)
    print("%s ..." % arr_str[:-2])
    print(
        "Min: %.2f, Max: %.2f, Avg: %.2f, Var: %.2f"
        % (np.min(arr), np.max(arr), np.mean(arr), np.var(arr))
    )


def load_model_and_set_outs(model_path, out_names):
    model = onnx.load(model_path)

    if out_names:

        for _ in range(len(model.graph.output)):
            model.graph.output.pop()

        for out_name in out_names:
            intermediate_tensor_name = out_name
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = intermediate_tensor_name
            model.graph.output.extend([intermediate_layer_value_info])

    else:
        pass

    f = io.BytesIO()
    onnx.save(model, f)
    f.seek(0)

    return f


def run_model(args):
    model_file = load_model_and_set_outs(args.model_path, args.outs)
    session = ort.InferenceSession(
        model_file.read(), providers=["CPUExecutionProvider"]
    )

    out_names = [i.name for i in session.get_outputs()]

    numpy_ins = np.load(args.ins_file)
    feeds = {k: numpy_ins.get(k.name) for k in session.get_inputs()}

    numpy_outs = session.run(out_names, feeds)

    for out_name, numpy_out in zip(out_names, numpy_outs):
        print("\033[92m[%s]\033[0m" % out_name)
        try:
            print_numpy_array(numpy_out)
        except:
            print(numpy_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ins-file", required=True, help="Numpy Npz File")
    parser.add_argument("--outs", nargs="+", type=str)
    args = parser.parse_args()
    run_model(args)
