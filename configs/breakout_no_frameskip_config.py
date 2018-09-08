import gym
import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor

from configs import EnvConfiguration
from utils.images import prepare_image
from wrappers.breakout_wrappers import MaxAndSkipEnv, NoopResetEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, \
    ClipRewardEnv
from wrappers.monitor_env import MonitorEnv
from wrappers.tensorboard_vec_env import TensorboardVecEnv
from wrappers.vec_env.dummy_vec_env import DummyVecEnv
from wrappers.vec_env.subproc_vec_env import SubprocVecEnv
from wrappers.vec_env.vec_frame_stack import VecFrameStack


#Config used by openAI baseline, useful to compare performance
class BreakoutNoFrameskipConfig(EnvConfiguration):

    def create_model(self, name, placeholders, reuse=False):
        with tf.variable_scope(name, reuse=reuse):

            previous_layer = prepare_image(placeholders['s0'])

            activ = tf.nn.relu
            previous_layer = tf.contrib.layers.conv2d(
                inputs=previous_layer,
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

            total_size = np.prod([v.value for v in previous_layer.get_shape()[1:]])
            previous_layer = tf.reshape(previous_layer, [-1, total_size])

            for idx in range(1):
                hidden_layer = tf.contrib.layers.fully_connected(
                    inputs=previous_layer,
                    num_outputs=512,
                    activation_fn=activ,
                    weights_initializer=tf.orthogonal_initializer(np.sqrt(2))
                )
                previous_layer = hidden_layer
            return previous_layer

    def _parameters(self):
        return {
            "seed": 1,
            "decay": False,
            "num_env": 8,
            "nb_steps": 128,
            "nb_epochs": 4,
            "nb_minibatch": 4,
            "clipping": 0.1,
            "learning_rate": 0.00025,
            "total_timesteps": int(80e6),
        }

    @property
    def env_name(self):
        return "BreakoutNoFrameskip-v4"

    def make_env(self, proc_idx=0, summary_path=None, renderer=False):
        env = gym.make(self.env_name)

        env.seed(self.parameters.seed + proc_idx)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = MonitorEnv(env)
        if summary_path:
            # Put Monitor before any wrappers that change the episode duration to get full episode in video
            env = Monitor(env, directory=summary_path + "/" + str(proc_idx), resume=True)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)

        return env

    def make_vec_env(self, summary_path=None, renderer=False):
        if renderer:
            venv = DummyVecEnv([self.make_env_fn()])
        else:
            venv = SubprocVecEnv([self.make_env_fn(i, summary_path) for i in range(self.parameters.num_env)])
            venv = TensorboardVecEnv(venv, summary_path)

        venv = VecFrameStack(venv, 4)
        return venv
