This repository is a reimplementation of the PPO algorithm that I made to get a better understanding of reinforcement learning and actor critic algorithms in particular.

It is based on the [ppo paper](https://arxiv.org/abs/1707.06347) and the accompanying github repository, [openAI baselines](https://github.com/openai/baselines)

I later implemented [curiosity-driven exploration](https://pathak22.github.io/noreward-rl/) as well.

### Requirements

My development environment was using python3.5, tensorflow-gpu 1.9, CUDA 9.0.

But it should also work with more recent versions and even without a GPU (though some environments like Breakout will be much slower)

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

This will start training the model with the configuration specified in `configs/breakout_no_frameskip_config.py`

You can see all the options with: `python3 my_ppo.py -h`

#### Inference
When you start the training, a window will also appear to show the agent playing. 

There are actually several agents training at the same time, what you see in this window is just a way to see how well your agent is currently playing.

You can toggle the presence of the window by pressing 'r'+'Enter' on your keyboard. 

#### Tensorboard

You can see the results of your algorithm by running:

`tensorboard --logdir=train`

The most interesting graph should be `Stats/TotalReward` which shows the average reward that the agent is getting per episode.

#### Videos
Another way to see how well the agents are doing is to look at videos taken during the training process.
You can find these videos in the `train` folder.

#### Saves

The latest version of the model is saved regularly. 

You can restart a training session from the saved model using the `--load` flag:

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

### Results for Breakout

This implementation of ppo produces results similar to those of openAI baselines repository:

OpenAI Baselines results for `BreakoutNoFrameSkip-v4` with `--num_timesteps=10000000 --nsteps=128`:

![alt text](https://github.com/LeSphax/ppo/raw/master/brag/OpenAI%20baselines.png "OpenAI baselines results for Breakout")

Results for `python3 my_ppo.py TestingTheRepository BreakoutNoFrameskip-v4`:

![alt text](https://github.com/LeSphax/ppo/raw/master/brag/BreakoutNoFrameskip-v4%2010M%20steps.png "MyPPO results for Breakout")


You can also see videos from [the beginning](https://github.com/LeSphax/ppo/raw/master/brag/BreakoutBefore.mp4) of the training and [the end](https://github.com/LeSphax/ppo/raw/master/brag/BreakoutAfter.mp4)

### Results for curiosity

For curiosity I didn't compare with a baseline, I just wanted to make sure that I implemented it correctly.

So I made sure that the agent could train using:
* Only intrinsic rewards and no extrinsic rewards (Red curve) 

`python3 my_ppo.py TestingTheRepository FixedButton-v0 --curiosity --norewards`
* Both intrinsic and extrinsic rewards (Blue curve)

`python3 my_ppo.py TestingTheRepository FixedButton-v0 --curiosity`

![alt text](https://github.com/LeSphax/ppo/raw/master/brag/FixedButtonEpisodeReward.png "FixedButton mean episodic reward")

As expected the agent does better when we add extrinsic rewards but it still learns to click the buttons when only intrinsic rewards are present.
This is because trying to predict where the next button will appear is harder than predicting where a click will appear.

One interesting finding was the need to clip actions before feeding them to the inverse predictor:
* In the FixedButton-v0 environment actions are clipped to 0,1 to make it impossible to click outside the screen. If the agent sends 30, 30 as an action, it will still click on position 1,1.
* But with a normal implementation of the algorithm, the actions used to train curiosity are not clipped.
* The problem is that the inverse predictor is trying to predict the action that was done given the current state and the previous state.
* If we don't clip actions, it won't be possible to do it. Because the actions 1,1 and 30,30 result in the same state.
* Since the agent gets rewards the inverse predictor is making mistakes it will exploit this mechanism and output only actions outside of the range of the environment.

So to fix this problem I added the clipping as part of the fixed_button_config. But this does not seem like a good solution, because this makes the algorithm dependent on the environment it is training in.