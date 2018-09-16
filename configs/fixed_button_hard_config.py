from configs.fixed_button_config import FixedButtonConfig


class FixedButtonHardConfig(FixedButtonConfig):

    @property
    def env_name(self):
        return "FixedButtonHard-v0"
