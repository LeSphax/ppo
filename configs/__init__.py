from abc import ABC, abstractmethod, abstractproperty, ABCMeta
from types import SimpleNamespace

registry = {}


def register_class(target_class):
    env_name = target_class().env_name
    if isinstance(env_name, list):
        for name in env_name:
            registry[name] = target_class
    else:
        registry[env_name] = target_class


def get_config(env_name):
    return registry[env_name]()


class EnvConfigRegistration(ABCMeta):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)

        if bases != (ABC,):
            register_class(cls)
        return cls


class EnvConfiguration(ABC, metaclass=EnvConfigRegistration):

    @abstractmethod
    def create_model(self, name, placeholders, reuse=False):
        """
        Creates the model to use for the policy and value function of this environment
        """
        pass

    @property
    def parameters(self):
        parameters = SimpleNamespace(**self._parameters())
        parameters.nb_updates = parameters.total_timesteps // parameters.nb_steps // parameters.num_env
        parameters.batch_size = parameters.nb_steps * parameters.num_env
        parameters.minibatch_size = parameters.batch_size // parameters.nb_minibatch
        return parameters

    @abstractmethod
    def _parameters(self):
        """
        A dictionary containing the ppo parameters
        """
        pass

    @abstractproperty
    def env_name(cls):
        """
        The name of the gym environment (i.e CartPole-v1)
        !!! This is an abstract class method, implement it with the @classmethod !!!
        """
        raise NotImplementedError

    @abstractmethod
    def make_env(self, proc_idx, summary_path, renderer=False):
        """
        Returns the environment
        """
        pass

    def make_env_fn(self, proc_idx=0, summary_path=None):
        def _thunk():
            return self.make_env(proc_idx, summary_path)

        return _thunk

    @abstractmethod
    def make_vec_env(self, summary_path):
        """
        Returns the environment
        """
        pass


from configs.breakout_config import *
from configs.breakout_no_frameskip_config import *
from configs.cartpole_config import *
from configs.vizdoom_config import *
from configs.random_button_config import *
