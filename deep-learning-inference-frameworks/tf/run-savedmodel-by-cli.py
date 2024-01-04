import numpy as np
import argparse
import subprocess


def run_savedmodel(args):
    arr_names = np.load(args.ins_file).files
    arr_cmd = []
    for arr_name in arr_names:
        arr_cmd.append("{0}={1}[{0}]".format(arr_name, args.ins_file))
    arr_cmd = ";".join(arr_cmd)

    subprocess.run(
        [
            "saved_model_cli",
            "run",
            "--dir",
            args.model_path,
            "--tag_set",
            "serve",
            "--signature_def",
            "serving_default",
            "--inputs",
            arr_cmd,
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ins-file", required=True)
    args = parser.parse_args()
    run_savedmodel(args)
