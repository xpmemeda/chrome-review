# NOTE:
# This script works well on onnx-1.20.1 and onnxruntime-1.23.2

import argparse
import onnx
from onnxruntime.transformers.optimizer import optimize_model
from collections import defaultdict, deque

from onnxruntime.transformers.onnx_model import OnnxModel

def fix_empty_outputs(model: onnx.ModelProto):
    # 先检查是否有人把 "" 当 input 用（极少见，但保险）
    has_empty_input = False
    for n in model.graph.node:
        if any(inp == "" for inp in n.input):
            has_empty_input = True
            print("WARNING: found empty-string input in node:", n.name, n.op_type)

    # 收集所有已用名字，避免冲突
    used = set()
    for n in model.graph.node:
        for x in n.input:
            if x: used.add(x)
        for x in n.output:
            if x: used.add(x)
    for t in model.graph.initializer:
        used.add(t.name)
    for v in model.graph.value_info:
        used.add(v.name)
    for v in model.graph.input:
        used.add(v.name)
    for v in model.graph.output:
        used.add(v.name)

    # 修复空输出
    counter = 0
    for i, n in enumerate(model.graph.node):
        for k, out in enumerate(n.output):
            if out == "":
                while True:
                    new_name = f"__optional_out_{i}_{k}_{counter}"
                    counter += 1
                    if new_name not in used:
                        break
                n.output[k] = new_name
                used.add(new_name)

    if has_empty_input:
        print("NOTE: model had empty inputs; if TRT still fails, we may need a full renaming map.")
    return model

def topo_sort_onnx_safe(model: onnx.ModelProto):
    graph = model.graph
    nodes = list(graph.node)
    num_nodes = len(nodes)

    producer = {}
    for idx, node in enumerate(nodes):
        for out in node.output:
            if not out:     # 忽略 ""
                continue
            if out in producer:
                raise RuntimeError(f"Duplicate output tensor name: {out} "
                                   f"(nodes {producer[out]} and {idx})")
            producer[out] = idx

    in_degree = [0] * num_nodes
    consumers = defaultdict(list)

    for idx, node in enumerate(nodes):
        for inp in node.input:
            if not inp:     # 忽略 ""
                continue
            if inp in producer:
                pred = producer[inp]
                consumers[pred].append(idx)
                in_degree[idx] += 1

    from collections import deque
    q = deque(i for i in range(num_nodes) if in_degree[i] == 0)
    sorted_indices = []

    while q:
        i = q.popleft()
        sorted_indices.append(i)
        for j in consumers[i]:
            in_degree[j] -= 1
            if in_degree[j] == 0:
                q.append(j)

    if len(sorted_indices) != num_nodes:
        raise RuntimeError("Graph has cycle or unresolved dependencies")

    del graph.node[:]
    for i in sorted_indices:
        graph.node.append(nodes[i])
    return model

def main(src_path, dst_path):
    m = OnnxModel(onnx.load(src_path))
    m.convert_float_to_float16(op_block_list=["RandomNormalLike"])

    m.model = fix_empty_outputs(m.model)
    m.model = topo_sort_onnx_safe(m.model)


    # onnx.checker.check_model(m.model, full_check=False)
    onnx.save_model(m.model, dst_path)


if __name__ == "__main__":
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("-s", "--src-path", type=str, required=True)
    cmd_parser.add_argument("-d", "--dst-path", type=str, required=True)
    cmd_arguments = cmd_parser.parse_args()

    main(cmd_arguments.src_path, cmd_arguments.dst_path)
