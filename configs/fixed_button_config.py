from configs import RandomButtonConfig


class FixedButtonConfig(RandomButtonConfig):

    @property
    def env_name(self):
        return "FixedButton-v0"
