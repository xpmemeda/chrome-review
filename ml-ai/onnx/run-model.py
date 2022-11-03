import numpy as np
import onnxruntime as ort
import argparse


def print_numpy_array(arr):
    print("arr: {}...".format(arr.reshape([-1]).tolist()[:6]))
    print(
        "min: {:.2f} max: {:.2f} avg: {:.2f} var: {:.2f}".format(
            np.min(arr), np.max(arr), np.mean(arr), np.var(arr)
        )
    )


def run_model(args):
    session = ort.InferenceSession(args.model_path, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in session.get_inputs()]
    out_names = [i.name for i in session.get_outputs()]

    ins_numpy = np.load(args.ins_file)
    feeds = {k: ins_numpy.get(k) for k in in_names}
    out_numpy = session.run(out_names, feeds)

    for name, arr in zip(out_names, out_numpy):
        print("\n[", name, "]")
        print_numpy_array(arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ins-file", required=True)
    args = parser.parse_args()
    run_model(args)
