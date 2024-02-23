import sys
import copy
import onnx
from onnx import helper

src_model_path = sys.argv[1]
tgt_model_path = sys.argv[2]

model = onnx.load(src_model_path)

graph = model.graph
initializers = {init.name for init in graph.initializer}

origin_inputs = copy.deepcopy(graph.input)
for _ in range(len(graph.input)):
    graph.input.pop()
for input in origin_inputs:
    if input.name not in initializers:
        graph.input.append(input)

onnx.save(model, tgt_model_path)
