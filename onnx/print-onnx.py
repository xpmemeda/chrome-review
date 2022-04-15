import onnx

model = onnx.load("/home/olafxiong/workspace/resources/elonsu/locallanzhou")
print(dir(model))
print(model.graph.name)
