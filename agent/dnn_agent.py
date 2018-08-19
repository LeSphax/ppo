import tensorflow as tf


class DNNAgent(object):

    def __init__(self, env, policy_class, value_class, model_function):
        input_shape = env.observation_space.shape
        output_size = env.action_space.n
        self.sess = tf.get_default_session()

        self.X, model_output_layer = model_function('shared_model', input_shape)

        self.LEARNING_RATE = tf.placeholder(tf.float32, (), name="learning_rate")
        self.CLIPPING = tf.placeholder(tf.float32, (), name="clipping")

        self.policy_estimator = policy_class(self.sess, self.X, model_output_layer, output_size=output_size, CLIPPING=self.CLIPPING)
        self.value_estimator = value_class(self.sess, self.X, model_output_layer, CLIPPING=self.CLIPPING)

        self.loss = self.policy_estimator.loss + 0.5 * self.value_estimator.loss

        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
        grads_and_vars = list(zip(grads, params))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE, epsilon=1e-5)
        self._train = optimizer.apply_gradients(grads_and_vars)

        self.sess = tf.get_default_session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def step(self, obs):
        return self.sess.run(
            [self.value_estimator.value, self.policy_estimator.action, self.policy_estimator.neglogp_action],
            {self.X: obs}
        )

    def get_action(self, obs):
        return self.sess.run(
            [self.policy_estimator.action, self.policy_estimator.neglogp_action],
            {self.X: obs}
        )

    def train(self, obs, values, actions, neglogp_actions, advantages, returns, clipping, learning_rate):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        entropy, loss, _ = self.sess.run(
            [self.policy_estimator.entropy, self.loss, self._train],
            {
                self.X: obs,
                self.policy_estimator.ACTIONS: actions,
                self.policy_estimator.OLDNEGLOGP_ACTIONS: neglogp_actions,
                self.policy_estimator.ADVANTAGES: advantages,
                self.value_estimator.OLD_VALUES: values,
                self.value_estimator.RETURNS: returns,
                self.CLIPPING: clipping,
                self.LEARNING_RATE: learning_rate,
            }
        )
        return entropy, loss

    def get_value(self, obs):
        return self.sess.run(self.value_estimator.value, {self.X: obs})
