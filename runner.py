import time

import numpy as np
import utils.tensorboard_util as tboard


class EnvRunner(object):
    def __init__(self, session, env, estimator, discount_factor=0.99, gae_weighting=0.95, **kwargs):
        self.sess = session
        self.estimator = estimator
        self.env = env
        self.obs = self.env.reset()
        self.discount_factor = discount_factor
        self.gae_weighting = gae_weighting
        self.kwargs = kwargs

    def run_timesteps(self, nb_timesteps):
        batch = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'bonuses': [],
            'values': [],
            'dones': [],
            'neglogp_actions': [],
            'advantages': [],
            'returns': [],
            'infos': []
        }
        epinfos = []

        for t in range(nb_timesteps):
            batch['obs'].append(self.obs)
            values, actions, neglogp_actions = self.estimator.step(self.obs)
            batch['values'].append(values)
            batch['neglogp_actions'].append(neglogp_actions)
            batch['actions'].append(actions)

            self.obs, rewards, dones, infos = self.env.step(actions)

            batch['rewards'].append(rewards)
            batch['dones'].append(dones)

            batch['infos'].append(infos)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

        advantages = np.zeros_like(batch['rewards'], dtype=float)

        batch['next_obs'] = batch['obs'][1:]
        batch['next_obs'].append(self.obs)

        if self.kwargs['use_rewards']:
            full_rewards = batch['rewards']
        else:
            full_rewards = np.zeros(np.shape(batch['rewards']))
        if self.kwargs['use_curiosity'] and self.estimator.curiosity:
            obs = flatten_venv(batch['obs'], swap=False)
            actions = flatten_venv(batch['actions'], swap=False)
            next_obs = flatten_venv(batch['next_obs'], swap=False)
            # tboard.add_image('Obs', obs[-1])
            bonus = self.estimator.get_bonus(obs, actions, next_obs)
            bonus = unflatten_venv(bonus, np.shape(batch['rewards']))
            batch['bonuses'] = bonus
            full_rewards += bonus
            tboard.add('Bonuses', bonus)

        last_values = self.estimator.get_value(self.obs)
        batch['values'].append(last_values)
        last_discounted_adv = np.zeros(self.env.num_envs)
        for idx in reversed(range(nb_timesteps)):
            use_future_rewards = 1 - batch['dones'][idx]
            next_value = batch['values'][idx + 1] * use_future_rewards

            td_error = self.discount_factor * next_value + full_rewards[idx] - batch['values'][idx]
            advantages[idx] = last_discounted_adv \
                = td_error + self.discount_factor * self.gae_weighting * last_discounted_adv * use_future_rewards

        batch['values'] = batch['values'][:-1]
        returns = advantages + batch['values']

        batch['advantages'] = advantages
        batch['returns'] = returns

        trajectory = {k: flatten_venv(batch[k]) for k in batch}
        return trajectory, epinfos


def flatten_venv(arr, swap=True):
    """
    swap and then flatten axes 0 and 1
    """
    arr = np.asarray(arr)
    s = arr.shape
    if swap:
        arr = arr.swapaxes(0, 1)
    return arr.reshape(s[0] * s[1], *s[2:])


def unflatten_venv(arr, original_shape):
    """
        unflatten an array that was passed through flatten_venv with swap to False
    """
    return arr.reshape(original_shape[0], original_shape[1], *original_shape[2:])
