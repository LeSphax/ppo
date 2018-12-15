import tensorflow as tf
from gym import spaces


class DNNAgent(object):

    def __init__(self, env, policy_class, value_class, config, use_curiosity):
        input_shape = env.observation_space.shape
        action_space = env.action_space
        self.use_curiosity = use_curiosity

        self.sess = tf.get_default_session()

        self.placeholders = {
            'learning_rate': tf.placeholder(tf.float32, (), name="learning_rate"),
            'clipping': tf.placeholder(tf.float32, (), name="clipping"),
            's0': tf.placeholder(shape=(None,) + input_shape, dtype=tf.float32, name="X"),
            's1': tf.placeholder(shape=(None,) + input_shape, dtype=tf.float32, name="X"),
            'advantages': tf.placeholder(tf.float32, [None]),
            'actions': tf.placeholder(dtype=action_space.dtype, shape=self.get_action_shape(action_space), name='Actions'),
            'oldneglogp_actions': tf.placeholder(tf.float32, [None]),
            'old_values': tf.placeholder(tf.float32, [None], name="old_values"),
            'returns': tf.placeholder(tf.float32, [None], name="returns"),
        }

        model_output_layer = config.create_model('shared_model', self.placeholders)

        self.policy_estimator = policy_class(self.sess, self.placeholders, model_output_layer, action_space=env.action_space)
        self.value_estimator = value_class(self.sess, self.placeholders, model_output_layer)

        self.loss = self.ppo_loss = self.policy_estimator.loss + 0.5 * self.value_estimator.loss

        if self.use_curiosity:
            if not hasattr(config, 'state_action_predictor'):
                raise AttributeError('The agent was told to use curiosity but no configuration was set for the curiosity model')
            self.curiosity = config.state_action_predictor(self.placeholders)
            self.loss = self.ppo_loss + self.curiosity['loss'] * 10

        params = tf.trainable_variables()
        grads = tf.gradients(self.loss, params)
        grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
        grads_and_vars = list(zip(grads, params))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.placeholders['learning_rate'], epsilon=1e-5)
        self._train = optimizer.apply_gradients(grads_and_vars)

        self.sess = tf.get_default_session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def step(self, obs):
        return self.sess.run(
            [self.value_estimator.value, self.policy_estimator.action, self.policy_estimator.neglogp_action],
            {self.placeholders['s0']: obs}
        )

    def get_action(self, obs):
        return self.sess.run(
            [self.policy_estimator.action, self.policy_estimator.neglogp_action],
            {self.placeholders['s0']: obs}
        )

    def train(self, *, obs, next_obs, values, actions, neglogp_actions, advantages, returns, clipping, learning_rate):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        run_list = [self.policy_estimator.entropy, self.loss,  self._train]
        if self.use_curiosity:
            run_list.extend([self.ppo_loss, self.curiosity['forward_loss'], self.curiosity['inverse_loss']])

        values = self.sess.run(
            run_list,
            {
                self.placeholders['s0']: obs,
                self.placeholders['s1']: next_obs,
                self.placeholders['actions']: actions,
                self.placeholders['oldneglogp_actions']: neglogp_actions,
                self.placeholders['advantages']: advantages,
                self.placeholders['old_values']: values,
                self.placeholders['returns']: returns,
                self.placeholders['clipping']: clipping,
                self.placeholders['learning_rate']: learning_rate,
            }
        )
        run_out = dict(zip(run_list, values))
        stats = {
            'Stats/Entropy': run_out[self.policy_estimator.entropy],
            'Stats/Loss': run_out[self.loss],
        }
        if self.use_curiosity:
            stats['Stats/PPO loss'] = run_out[self.ppo_loss]
            stats['Stats/Forward loss'] = run_out[self.curiosity['forward_loss']]
            stats['Stats/Inverse loss'] = run_out[self.curiosity['inverse_loss']]

        return stats

    def get_value(self, obs):
        return self.sess.run(self.value_estimator.value, {self.placeholders['s0']: obs})

    def get_bonus(self, obs, actions, next_obs):
        return self.curiosity['bonus_function'](self.sess, s0=obs, actions=actions, s1=next_obs)

    def get_action_shape(self, action_space):
        if isinstance(action_space, spaces.Discrete):
            shape = [None]
        elif isinstance(action_space, spaces.Box):
            size = action_space.shape[0]
            shape = [None, size]
        else:
            raise NotImplementedError
        return shape
