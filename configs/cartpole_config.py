import numpy as np
import tensorflow as tf

import gym
from wrappers.monitor_env import MonitorEnv
from wrappers.tensorboard_vec_env import TensorboardVecEnv
from wrappers.vec_env.dummy_vec_env import DummyVecEnv
from wrappers.vec_env.subproc_vec_env import SubprocVecEnv
from wrappers.vec_env.vec_normalize import VecNormalize
from configs import EnvConfiguration


class CartPoleConfig(EnvConfiguration):

    def create_model(self, name, placeholders, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            previous_layer = placeholders['s0']

            for idx in range(2):
                hidden_layer = tf.contrib.layers.fully_connected(
                    inputs=previous_layer,
                    num_outputs=64,
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
                )
                previous_layer = hidden_layer
            return previous_layer

    def _parameters(self):
        return {
            "seed": 1,
            "decay": True,
            "num_env": 8,
            "nb_steps": 64,
            "nb_epochs": 4,
            "nb_minibatch": 4,
            "clipping": 0.1,
            "learning_rate": 2.5e-4,
            "total_timesteps": 1000000,
        }

    @property
    def env_name(cls):
        return "CartPole-v1"

    def make_env(self, proc_idx=0, summary_path=None, renderer=False):
        env = gym.make(self.env_name)

        env.seed(self.parameters.seed)

        env = MonitorEnv(env)

        return env

    def make_vec_env(self, summary_path=None, renderer=False):
        if renderer:
            venv = DummyVecEnv([self.make_env_fn()])
            venv = VecNormalize(venv, reuse=True)

        else:
            venv = SubprocVecEnv([self.make_env_fn(i, summary_path) for i in range(self.parameters.num_env)])
            venv = TensorboardVecEnv(venv)
            venv = VecNormalize(venv)

        return venv
