import argparse
import subprocess
import numpy as np


def str_to_numpy_dtype(type_str):
    if type_str == "DT_INT32":
        return np.int32
    if type_str == "DT_FLOAT":
        return np.float
    raise NotImplementedError("invalid type-str: " + type_str)


def generate_random_ins(args):
    savedmodelcli_cmd = [
        "saved_model_cli",
        "show",
        "--signature_def",
        "serving_default",
        "--tag_set",
        "serve",
        "--dir",
        args.model_path,
    ]
    process = subprocess.Popen(
        savedmodelcli_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = process.communicate()
    errcode = process.returncode
    if errcode:
        print(err)
        return

    import re

    pattern_inp_name = re.compile(r"""inputs\['\S+'\]""")
    inp_names = pattern_inp_name.findall(str(out))
    inp_names = [name[8:-2] for name in inp_names]
    pattern_oup_name = re.compile(r"""outputs\['\S+'\]""")
    oup_names = pattern_oup_name.findall(str(out))
    oup_names = [name[9:-2] for name in oup_names]

    pattern_dtype = re.compile(r"""dtype: \S+\\n""")
    dtypes = pattern_dtype.findall(str(out))
    dtypes = [dtype[7:-2] for dtype in dtypes]

    pattern_shape = re.compile(r"""shape: \(.+?\)""")
    shapes = pattern_shape.findall(str(out))
    shapes = [list(map(int, shape[8:-1].split(", "))) for shape in shapes]

    for shape in shapes:
        for idx in range(len(shape)):
            shape[idx] = 1 if shape[idx] == -1 else shape[idx]

    assert len(inp_names) + len(oup_names) == len(dtypes) == len(shapes)
    arrs = {}
    for name, dtype, shape in zip(inp_names, dtypes, shapes):
        numpy_dtype = str_to_numpy_dtype(dtype)
        arrs[name] = np.random.randint(0, 256, size=shape).astype(numpy_dtype)
    np.savez(args.out_path, **arrs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-path", required=True)
    args = parser.parse_args()
    generate_random_ins(args)
