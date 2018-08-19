import tensorflow as tf
import numpy as np


class CategoricalPd(object):
    def __init__(self, logits):
        self.logits = logits

    def neglogp(self, x):
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_actions)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)


class DNNPolicy(object):

    def __init__(self, sess, X, model_output_layer, output_size, CLIPPING, *, reuse=False):
        self.sess = sess
        self.X = X
        self.CLIPPING = CLIPPING

        with tf.variable_scope('policy', reuse=reuse):
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=model_output_layer,
                num_outputs=output_size,
                activation_fn=None,
                weights_initializer=tf.constant_initializer(0.01)
            )

            self.probability_distribution = CategoricalPd(self.output_layer)

            self.action = self.probability_distribution.sample()
            self.neglogp_action = self.probability_distribution.neglogp(self.action)

            self.ADVANTAGES = tf.placeholder(tf.float32, [None])
            self.ACTIONS = tf.placeholder(tf.int32, [None])
            self.OLDNEGLOGP_ACTIONS = tf.placeholder(tf.float32, [None])

            self.new_neglogp_action = self.probability_distribution.neglogp(self.ACTIONS)

            self.entropy = tf.reduce_mean(self.probability_distribution.entropy())

            ratio = tf.exp(self.OLDNEGLOGP_ACTIONS - self.new_neglogp_action)
            pg_losses = -self.ADVANTAGES * ratio
            pg_losses2 = -self.ADVANTAGES * tf.clip_by_value(ratio, 1.0 - self.CLIPPING, 1.0 + self.CLIPPING)
            self.loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2)) - self.entropy * 0.01
