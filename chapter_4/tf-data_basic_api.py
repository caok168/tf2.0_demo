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

dataset = tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)

for item in dataset:
    print(item)

# 1. repeat epoch
# 2. get batch
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

# interleave:
# case: 文件dataset -> 具体数据集

dataset2 = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v), # map_fn
    cycle_length = 5, # cycle_length
    block_length = 5, # block_length
)
for item in dataset2:
    print(item)

x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array(['cat', 'dog', 'fox'])
dataset3 = tf.data.Dataset.from_tensor_slices((x, y))
print(dataset3)

for item_x, item_y in dataset3:
    print(item_x.numpy(), item_y.numpy())

dataset4 = tf.data.Dataset.from_tensor_slices({"feature": x,
                                               "label": y})
for item in dataset4:
    print(item["feature"].numpy(), item["label"].numpy())