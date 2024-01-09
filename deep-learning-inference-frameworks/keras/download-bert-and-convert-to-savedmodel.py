import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os

os.environ["KERAS_HOME"] = os.path.join(os.environ["HOME"], ".keras")

import numpy as np
import keras_nlp
import keras

r"""
https://keras.io/api/keras_nlp/models/bert/bert_classifier/#bertclassifier-class
"""

features = ["The quick brown fox jumped.", "I forgot my homework."]
labels = [0, 3]

# Pretrained classifier.
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased",
    num_classes=4,
)

r = classifier.predict(x=features, batch_size=2)
print(r)

classifier.fit(x=features, y=labels, batch_size=2)
r = classifier.predict(x=features, batch_size=2)
print(r)

# Re-compile (e.g., with a new learning rate).
classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
    jit_compile=True,
)
# Access backbone programmatically (e.g., to change `trainable`).
classifier.backbone.trainable = False
# Fit again.
r = classifier.fit(x=features, y=labels, batch_size=2)
print(r)

import tensorflow as tf

tf.saved_model.save(classifier, "bert-classifier")
