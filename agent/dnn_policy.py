import tensorflow as tf
import numpy as np
from gym import spaces


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


class DiagGaussianPd(object):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape) - 1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))


class DNNPolicy(object):

    def __init__(self, sess, X, model_output_layer, action_space, CLIPPING, *, reuse=False):
        self.sess = sess
        self.X = X
        self.CLIPPING = CLIPPING

        with tf.variable_scope('policy', reuse=reuse):

            if isinstance(action_space, spaces.Discrete):
                output_layer = tf.contrib.layers.fully_connected(
                    inputs=model_output_layer,
                    num_outputs=action_space.n,
                    activation_fn=None,
                    weights_initializer=tf.constant_initializer(0.01)
                )
                self.probability_distribution = CategoricalPd(output_layer)
                shape = [None]
            elif isinstance(action_space, spaces.Box):
                size = action_space.shape[0]
                mean = tf.contrib.layers.fully_connected(
                    inputs=model_output_layer,
                    num_outputs=size,
                    activation_fn=None,
                    weights_initializer=tf.constant_initializer(0.01)
                )
                logstd = tf.get_variable(name='logstd', shape=[1, size], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                self.probability_distribution = DiagGaussianPd(pdparam)
                shape = [None, size]

            self.action = self.probability_distribution.sample()
            self.neglogp_action = self.probability_distribution.neglogp(self.action)

            self.ADVANTAGES = tf.placeholder(tf.float32, [None])
            self.ACTIONS = tf.placeholder(dtype=action_space.dtype, shape=shape, name='Actions')
            self.OLDNEGLOGP_ACTIONS = tf.placeholder(tf.float32, [None])

            self.new_neglogp_action = self.probability_distribution.neglogp(self.ACTIONS)

            self.entropy = tf.reduce_mean(self.probability_distribution.entropy())

            ratio = tf.exp(self.OLDNEGLOGP_ACTIONS - self.new_neglogp_action)
            pg_losses = -self.ADVANTAGES * ratio
            pg_losses2 = -self.ADVANTAGES * tf.clip_by_value(ratio, 1.0 - self.CLIPPING, 1.0 + self.CLIPPING)
            self.loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2)) - self.entropy * 0.01
