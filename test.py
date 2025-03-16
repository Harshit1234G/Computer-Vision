import tensorflow as tf
import numpy as np


a = tf.constant([1, 2, 1, 2])
print(np.tile(a, (3, 1)))
print(tf.transpose(tf.repeat(tf.expand_dims(a, axis= 1), 3, axis= 1)))
print(tf.broadcast_to(a, (3, 4)))