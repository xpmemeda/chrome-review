import typing as ty
import numpy as np
import onnxruntime as ort
import argparse


def run_model(args):
    session = ort.InferenceSession(args.onnx_model, providers=["CUDAExecutionProvider"])

    out_names = [i.name for i in session.get_outputs()]

    numpy_ins = np.load(args.data)
    feeds = {k.name: numpy_ins.get(k.name) for k in session.get_inputs()}

    numpy_outs = session.run(out_names, feeds)

    for out_name, numpy_out in zip(out_names, numpy_outs):
        print(
            "\033[92m[%s: %s %s]\033[0m" % (out_name, numpy_out.shape, numpy_out.dtype)
        )
        print(numpy_out)

    if args.output:
        npz: ty.Dict[str, np.ndarray] = {
            name: array for name, array in zip(out_names, numpy_outs)
        }
        np.savez(args.output, **npz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--onnx-model", type=str, required=True)
    parser.add_argument("-d", "--data", required=True, help="Npz format.")
    parser.add_argument(
        "-o", "--output", type=str, help="The path to dump result if specified."
    )
    args = parser.parse_args()
    run_model(args)
