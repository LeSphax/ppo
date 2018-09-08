import tensorflow as tf
import numpy as np


class DNNValue(object):

    def __init__(self, sess, placeholders, model_output_layer, *, reuse=False):
        self.sess = sess

        with tf.variable_scope('value', reuse=reuse):
            self.value = tf.contrib.layers.fully_connected(
                inputs=model_output_layer,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.orthogonal_initializer(1)
            )[:, 0]

            value_clipped = placeholders['old_values'] + tf.clip_by_value(self.value - placeholders['old_values'], -placeholders['clipping'],
                                                               placeholders['clipping'])
            losses1 = tf.square(self.value - placeholders['returns'])
            losses2 = tf.square(value_clipped - placeholders['returns'])
            self.loss = tf.reduce_mean(tf.maximum(losses1, losses2))
