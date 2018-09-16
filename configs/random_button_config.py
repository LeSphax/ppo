import gym
import gym_ui
import numpy as np
import tensorflow as tf
import utils.tensorboard_util as tboard
from gym.wrappers import TimeLimit

from configs import EnvConfiguration
from utils.images import prepare_image
from wrappers.breakout_wrappers import WarpFrame
from wrappers.monitor_env import MonitorEnv
from wrappers.tensorboard_vec_env import TensorboardVecEnv
from wrappers.vec_env.dummy_vec_env import DummyVecEnv
from wrappers.vec_env.subproc_vec_env import SubprocVecEnv


class RandomButtonConfig(EnvConfiguration):

    def curiosity_encoder(self, state, reuse=False):
        with tf.variable_scope('curiosity', reuse=reuse):
            prepared_image = prepare_image(state)

            activ = tf.nn.elu
            previous_layer = tf.contrib.layers.conv2d(
                inputs=prepared_image,
                num_outputs=32,
                kernel_size=8,
                padding="valid",
                activation_fn=activ,
                stride=4,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )

            previous_layer = tf.contrib.layers.conv2d(
                inputs=previous_layer,
                num_outputs=64,
                kernel_size=4,
                padding="valid",
                activation_fn=activ,
                stride=2,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )

            previous_layer = tf.contrib.layers.conv2d(
                inputs=previous_layer,
                num_outputs=64,
                kernel_size=3,
                padding="valid",
                activation_fn=activ,
                stride=1,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )

            previous_layer = tf.reshape(previous_layer, [-1, np.prod(previous_layer.get_shape().as_list()[1:])])

            return tf.contrib.layers.fully_connected(
                inputs=previous_layer,
                num_outputs=32,
                activation_fn=activ,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )

    def state_action_predictor(self, placeholders):
        clipped_actions = tf.clip_by_value(placeholders['actions'],0,1)
        encoder_s0 = self.curiosity_encoder(placeholders['s0'])
        encoder_s1 = self.curiosity_encoder(placeholders['s1'], reuse=True)
        # encoder_s0 = tf.print(encoder_s0, [encoder_s0], summarize=16)

        activ = tf.nn.elu

        inverse_encoding = tf.concat([encoder_s0, encoder_s1], 1)
        inverse_fc = tf.contrib.layers.fully_connected(
            inputs=inverse_encoding,
            num_outputs=64,
            activation_fn=activ,
            weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
        )
        inverse_logits = tf.contrib.layers.fully_connected(
            inputs=inverse_fc,
            num_outputs=2,
            activation_fn=activ,
            weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
        )

        # inverse_logits = tf.print(inverse_logits, [inverse_logits], summarize=16)

        inverse_loss = tf.reduce_mean(tf.square(tf.subtract(inverse_logits, clipped_actions)), name="invloss")

        forward_encoding = tf.concat([encoder_s0, clipped_actions], 1)
        forward_fc = tf.contrib.layers.fully_connected(
            inputs=forward_encoding,
            num_outputs=64,
            activation_fn=activ,
            weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
        )
        forward_next_state = tf.contrib.layers.fully_connected(
            inputs=forward_fc,
            num_outputs=encoder_s0.get_shape()[1].value,
            activation_fn=activ,
            weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
        )
        # encoder_s1 = tf.print(encoder_s1, [encoder_s1], summarize=100)

        forward_losses = tf.square(tf.subtract(forward_next_state, encoder_s1))
        forward_losses = tf.reduce_mean(forward_losses, 1, name="forwardlosses")  # Reduce only the encoding dimensions to allow batch inferences

        forward_loss = 0.5 * tf.reduce_mean(forward_losses, name="forwardloss")
        forward_loss = encoder_s0.get_shape()[1].value * forward_loss  # make loss independent of output_size

        def get_bonus(sess, s0, actions, s1):
            error, predicted_next_state = sess.run([forward_losses, forward_next_state], {placeholders['s0']: s0, placeholders['s1']: s1, placeholders['actions']: actions})
            # tboard.add_image('Predicted next', predicted_next_state[-1])
            return error

        return {'inverse_loss': inverse_loss, 'forward_loss': forward_loss, 'loss': inverse_loss * 0.8 + forward_loss * 0.2, 'bonus_function': get_bonus}

    def create_model(self, name, placeholders, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            prepared_image = prepare_image(placeholders['s0'])

            activ = tf.nn.elu
            previous_layer = tf.contrib.layers.conv2d(
                inputs=prepared_image,
                num_outputs=4,
                kernel_size=4,
                padding="valid",
                activation_fn=activ,
                stride=2,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )

            previous_layer = tf.contrib.layers.conv2d(
                inputs=previous_layer,
                num_outputs=4,
                kernel_size=3,
                padding="valid",
                activation_fn=activ,
                stride=1,
                weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
            )

            total_size = np.prod([v.value for v in previous_layer.get_shape()[1:]])
            previous_layer = tf.reshape(previous_layer, [-1, total_size])

            for idx in range(1):
                hidden_layer = tf.contrib.layers.fully_connected(
                    inputs=previous_layer,
                    num_outputs=16,
                    activation_fn=activ,
                    weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
                )
                previous_layer = hidden_layer
            return previous_layer

    def _parameters(self):
        return {
            "seed": 1,
            "decay": False,
            "num_env": 16,
            "nb_steps": 128,
            "nb_epochs": 4,
            "nb_minibatch": 4,
            "clipping": 0.1,
            "learning_rate": 0.00025,
            "total_timesteps": int(1e6),
        }

    @property
    def env_name(self):
        return "RandomButton-v0"

    def make_env(self, proc_idx=0, summary_path=None, renderer=False):
        env = gym.make(self.env_name)

        env.seed(self.parameters.seed + proc_idx)
        env = TimeLimit(env, max_episode_steps=100)
        env = MonitorEnv(env)
        # if summary_path:
        #     # Put Monitor before any wrappers that change the episode duration to get full episode in video
        #     env = Monitor(env, directory=summary_path + "/" + str(proc_idx), resume=True)
        env = WarpFrame(env, size=64)

        return env

    def make_vec_env(self, summary_path=None, renderer=False):
        if renderer:
            venv = DummyVecEnv([self.make_env_fn()])
        else:
            venv = SubprocVecEnv([self.make_env_fn(i, summary_path) for i in range(self.parameters.num_env)])
            venv = TensorboardVecEnv(venv)

        # venv = VecFrameStack(venv, 4)
        return venv
