import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)


loaded_keras_model = keras.models.load_model('./graph_def_and_weights/fashion_mnist_model.h5')
loaded_keras_model(np.ones((1, 28, 28)))

run_model = tf.function(lambda x : loaded_keras_model(x))
keras_concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(loaded_keras_model.inputs[0].shape,
                  loaded_keras_model.inputs[0].dtype))

keras_concrete_func(tf.constant(np.ones((1, 28, 28), dtype=np.float32)))
