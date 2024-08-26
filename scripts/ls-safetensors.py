import os
import argparse

from safetensors.torch import safe_open


def main(cmd_arguments):
    model_dir = cmd_arguments.d

    all_keys = set()
    for fname in os.listdir(model_dir):
        if fname.endswith(".safetensors"):
            path = os.path.join(model_dir, fname)
            with safe_open(path, framework="pt") as f:
                if cmd_arguments.detail:
                    for k in f.keys():
                        t = f.get_tensor(k)
                        all_keys.add((k, t.shape, t.dtype))
                else:
                    all_keys.update(f.keys())

    for key in sorted(all_keys):
        print(key)


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--d", type=str, help="directory", required=True)
    cmd_parser.add_argument("--detail", action="store_true")
    cmd_arguments = cmd_parser.parse_args()
    main(cmd_arguments)
