import os
import tempfile

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

tmpdir = tempfile.mkdtemp()
tmpdir = os.path.dirname(os.path.abspath(__file__))
physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

file = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg",
)
img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
plt.imshow(img)
plt.axis("off")
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis, ...])
np.save("x.npy", x)
# type(x): <class 'numpy.ndarray'>

labels_path = tf.keras.utils.get_file(
    "ImageNetLabels.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
)
imagenet_labels = np.array(open(labels_path).read().splitlines())


pretrained_model = tf.keras.applications.MobileNet()
print(type(tf.keras.applications.MobileNet))  # <class 'function'>
print(type(pretrained_model))  # <class 'keras.engine.functional.Functional'>
result_before_save = pretrained_model(x)

decoded = imagenet_labels[np.argsort(result_before_save)[0, ::-1][:5] + 1]

print("Result before saving:\n", decoded)


mobilenet_save_path = os.path.join(tmpdir, "mobilenet/1/")
tf.saved_model.save(pretrained_model, mobilenet_save_path)

# <class 'tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject'>
loaded = tf.saved_model.load(mobilenet_save_path)
print(type(loaded))
print(list(loaded.signatures.keys()))  # ["serving_default"]

infer = loaded.signatures["serving_default"]
print(type(infer))  # <class 'tensorflow.python.saved_model.load._WrapperFunction'>
print(infer.structured_outputs)

labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]

decoded = imagenet_labels[np.argsort(labeling)[0, ::-1][:5] + 1]

print("Result after saving and loading:\n", decoded)
