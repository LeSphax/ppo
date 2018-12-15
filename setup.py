from setuptools import setup

setup(
    name='ppo',
    version='',
    packages=['', 'utils', 'configs', 'wrappers', 'wrappers.vec_env'],
    url='',
    license='',
    author='Sebastien Kerbrat',
    author_email='sbkerbrat@gmail.com',
    description='A reimplementation of the ppo algorithm based on openAI baselines',
    install_requires=[
        'docopt',
        'gym[atari,classic_control]',
        'matplotlib',
        'numpy',
        'opencv-python',
        'six',
        'tensorboard',
        'tensorflow',
    ]
)
