import utils.tensorboard_util as tboard

from wrappers.vec_env import VecEnvWrapper


class TensorboardVecEnv(VecEnvWrapper):
    def __init__(self, venv, *, reward_key='total_reward'):
        VecEnvWrapper.__init__(self, venv=venv)
        self.reward_key = reward_key
        self.epinfos = []

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo: tboard.add('total_reward', maybeepinfo[self.reward_key])

        return obs, rews, news, infos

    def reset(self):
        return self.venv.reset()
