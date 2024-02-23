import sys
import onnx

model = onnx.load(sys.argv[1])
opset_version = model.opset_import[0].version if len(model.opset_import) > 0 else None

print(opset_version)
