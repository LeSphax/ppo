This repository is a reimplementation of the PPO algorithm that I made to get a better understanding of reinforcement learning and actor critic algorithms in particular.

It is based on the [ppo paper](https://arxiv.org/abs/1707.06347) and the accompanying github repository, [openAI baselines](https://github.com/openai/baselines)

I later implemented [curiosity-driven exploration](https://pathak22.github.io/noreward-rl/) as well.

### Requirements

My development environment was using python3.5, tensorflow-gpu 1.9, CUDA 9.0.
But it should also work with more recent versions and even without a GPU (though some environments like breakout will be much slower)

### Installation

Clone the repository and install the dependencies: 
```
git clone git@github.com:LeSphax/ppo.git
cd ppo
pip3 install -e .
```

### Training

Run the algorithm on one of the configured environments:
`python3 my_ppo.py (<label>) (<env_name>) [options]`

So for example, to run the environment BreakoutNoFrameskip-v4, you would type:
`python3 my_ppo.py TestingTheRepository BreakoutNoFrameskip-v4`

This will start training the model with the configuration specified in configs/breakout_no_frameskip_config.py

You can see all the options with: `python3 my_ppo.py -h`

#### Inference
When you start the training, a window will also appear to show the agent playing. 

There are actually several agents training at the same time, what you see in this window is just a way to see how well your agent is currently playing.

You can toggle the presence of the window by pressing 'r'+'Enter' on your keyboard. 

#### Tensorboard

You can see the results of your algorithm by running:

`tensorboard --logdir=train`

The most interesting graph should be Stats/TotalReward which shows the average reward that the agent is getting per episode.

#### Saves

The latest version of the model is saved regularly. 

You can restart a training session from the saved model using the --load flag:

`python3 my_ppo.py TestingTheRepository BreakoutNoFrameskip-v4 --load`

#### Configurations

The structure of the neural network and the hyperparameters to use for a specific environment is located in the config files. You can find them in the `configs` folder.

Configurations exist for the following gym environments:
* Breakout-v0
* BreakoutNoFrameskip-v4
* CartPole-v1

And the [following environments](https://github.com/LeSphax/gym-ui) that I created myself to test the curiosity-driven exploration algorithm:
* FixedButton-v0
* FixedButtonHard-v0
* RandomButton-v0

#### Curiosity 

To train using curiosity you can use the `--curiosity` flag, you can also use the `--norewards` flag to ignore the extrinsic rewards that the environment is giving.
To support curiosity-driven exploration a config must implement the `state_action_predictor` and `curiosity_encoder` methods.

### Results

This implementation of ppo produces results similar to those of openAI baselines repository:

Baselines results for `BreakoutNoFrameSkip-v4` with `--num_timesteps=10000000 --nsteps=128`:

![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

//My graph with legend
Baselines vs