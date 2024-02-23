import onnx
import argparse


def main(src_path: str, dst_path: str):
    model = onnx.load(src_path)
    onnx.save_model(model, dst_path, save_as_external_data=False)


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("-s", "--src-model-path", type=str, required=True)
    cmd_parser.add_argument("-d", "--dst-model-path", type=str, required=True)
    cmd_arguments = cmd_parser.parse_args()

    main(cmd_arguments.src_model_path, cmd_arguments.dst_model_path)
