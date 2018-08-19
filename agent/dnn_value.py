import tensorflow as tf
import numpy as np


class DNNValue(object):

    def __init__(self, sess, X, model_output_layer, CLIPPING, *, reuse=False):
        self.sess = sess
        self.X = X
        self.CLIPPING = CLIPPING

        with tf.variable_scope('value', reuse=reuse):
            self.value = tf.contrib.layers.fully_connected(
                inputs=model_output_layer,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.orthogonal_initializer(1)
            )[:, 0]

            self.OLD_VALUES = tf.placeholder(tf.float32, [None], name="old_values")
            self.RETURNS = tf.placeholder(tf.float32, [None], name="returns")

            value_clipped = self.OLD_VALUES + tf.clip_by_value(self.value - self.OLD_VALUES, -self.CLIPPING,
                                                               self.CLIPPING)
            losses1 = tf.square(self.value - self.RETURNS)
            losses2 = tf.square(value_clipped - self.RETURNS)
            self.loss = tf.reduce_mean(tf.maximum(losses1, losses2))
