# NOTE:
# This script works well on tensorrt (8.6.1.post1 ~ 10.9.0.32)

import os
import pdb
import argparse
import tensorrt as trt
import typing as ty
import pycuda.driver as cuda
import pycuda.autoinit  # NOTE: import this module to init cuda.
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine_from_onnx(onnx_path: str, engine_file_path: str, quant: str):
    if os.path.exists(engine_file_path):
        print(f"Engine file already found at {engine_file_path}. Skipping build.")
        return

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if quant == "f16":
            config.set_flag(trt.BuilderFlag.FP16)

        cwd = os.getcwd()
        os.chdir(os.path.dirname(cmd_arguments.onnx_model))

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        os.chdir(cwd)

        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        profile.set_shape(
            input_name,
            min=(1, 3, 224, 224),
            opt=(1, 3, 224, 224),
            max=(1, 3, 224, 224),
        )
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)

        with open(engine_file_path, "wb") as f:
            print(f"Saving serialized engine to {engine_file_path}...")
            f.write(serialized_engine)
        print("Engine saved successfully.")


def load_engine(engine_file_path) -> trt.ICudaEngine:
    print(f"Loading engine from {engine_file_path}...")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        if engine is None:
            raise RuntimeError("Failed to deserialize the engine.")
        print("Engine loaded successfully.")
        return engine


def allocate_buffers(engine):
    inputs = []
    outputs = []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)

        tensor_shape = engine.get_tensor_shape(tensor_name)
        size = trt.volume(tensor_shape)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        host_mem: np.ndarray = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(
                {
                    "host": host_mem,
                    "device": device_mem,
                    "name": tensor_name,
                    "shape": tensor_shape,
                }
            )
        else:
            outputs.append(
                {
                    "host": host_mem,
                    "device": device_mem,
                    "name": tensor_name,
                    "shape": tensor_shape,
                }
            )

    return inputs, outputs, stream


def infer(engine, data: ty.Dict[str, np.ndarray]):
    context = engine.create_execution_context()

    inputs, outputs, stream = allocate_buffers(engine)

    print("inps", inputs)
    print("outs", outputs)

    # 1. inps h2d.
    for inp in inputs:
        np.copyto(inp["host"], data[inp["name"]].ravel())
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    # 2. infer.
    for inp in inputs:
        context.set_tensor_address(inp["name"], int(inp["device"]))
    for out in outputs:
        context.set_tensor_address(out["name"], int(out["device"]))
    context.execute_async_v3(stream.handle)

    # 3. outs d2h.
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

    stream.synchronize()

    result = {}
    for out in outputs:
        output_shape = context.get_tensor_shape(out["name"])
        output_data = out["host"].reshape(output_shape)
        result[out["name"]] = output_data

    return result


if __name__ == "__main__":
    cmd_arguments_parser = argparse.ArgumentParser()
    cmd_arguments_parser.add_argument("-m", "--onnx-model", type=str, required=True)
    cmd_arguments_parser.add_argument(
        "-e",
        "--engine",
        type=str,
        required=True,
        help="Path to save trt compiled engine, if file existed, load it directly and will not use onnx model.",
    )
    cmd_arguments_parser.add_argument(
        "-q",
        "--quant",
        type=str,
        choices=["f16", "f32"],
        required=True,
    )
    cmd_arguments_parser.add_argument(
        "-d", "--data", type=str, required=True, help="Npz format."
    )
    cmd_arguments_parser.add_argument(
        "-o", "--output", type=str, help="The path to dump result if specified."
    )
    cmd_arguments = cmd_arguments_parser.parse_args()

    data = np.load(cmd_arguments.data)
    build_engine_from_onnx(
        cmd_arguments.onnx_model, cmd_arguments.engine, cmd_arguments.quant
    )
    engine = load_engine(cmd_arguments.engine)
    outs = infer(engine, data)

    for k, v in outs.items():
        print("\033[92m[%s: %s %s]\033[0m" % (k, v.shape, v.dtype))
        print(v)

    if cmd_arguments.output:
        np.savez(cmd_arguments.output, **outs)
