import numpy as np
import sys
import time
import onnxruntime as ort

print(ort.get_device())

def main():
    # sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options = None

    model = sys.argv[1]
    session = ort.InferenceSession(model, sess_options=sess_options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    inputs = [i.name for i in session.get_inputs()]
    outputs = [i.name for i in session.get_outputs()]
    print(inputs)
    print(outputs)
    data = np.load(sys.argv[2])
    feeds = {}
    for k in inputs:
        feeds[k] = data.get(k)
    for i in range(5):
        result = session.run(outputs, feeds)
    t1 = time.time()
    for i in range(20):
        result = session.run(outputs, feeds)
    print('[time] ', time.time() - t1)
    for name, value in zip(outputs, result):
        print('\n[', name, ']')
        print('sum: ', np.sum(value))
        print('avg: ', np.mean(value))
        print('var: ', np.var(value))
        print('max: ', np.max(value))
        print('min: ', np.min(value))
        print(value)

if __name__ == "__main__":
    main()
