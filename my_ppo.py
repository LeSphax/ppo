#!/usr/bin/env python3
import _thread
import os

import configs
from configs import *
from wrappers.sample_runner import SampleRunner

if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from agent.dnn_agent import DNNAgent

from runner import EnvRunner
from agent.dnn_policy import DNNPolicy
from agent.dnn_value import DNNValue

import numpy as np
import utils.keyPoller as kp
import utils.tensorboard_util as tboard
from datetime import datetime
import time
import multiprocessing
from docopt import docopt
import tensorflow as tf

_USAGE = '''
Usage:
    my_ppo (<label>) (<env_name>) [--debug|--load]
    
Options:
    --debug                        Tensorflow debugger
    --load                        Load the last save with this name

'''
options = docopt(_USAGE)

label = str(options['<label>'])
env_name = str(options['<env_name>'])
load = options['--load']

date = datetime.now().strftime("%m%d-%H%M")



def simulate():
    config = configs.get_config(env_name)
    parameters = config.parameters

    save_dir = os.path.join(os.getcwd(), 'save/{env_name}/{label}'.format(env_name=config.env_name, label=label))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'latest_save')

    summary_path = os.path.join(os.getcwd(), 'train/{env_name}/{label}_{date}'.format(env_name=config.env_name, label=label, date=date))
    tboard.init(summary_path)

    def make_session():
        ncpu = multiprocessing.cpu_count()
        if sys.platform == 'darwin':
            ncpu //= 2
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=ncpu,
                                inter_op_parallelism_threads=ncpu)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.__enter__()

    make_session()
    sess = tf.get_default_session()

    venv = config.make_vec_env(summary_path)
    estimator = DNNAgent(venv, policy_class=DNNPolicy, value_class=DNNValue, config=config)
    saver = tf.train.Saver()
    if load:
        tf.train.Saver().restore(sess, save_path)

    def renderer_thread(estimator, sess):
        with sess.as_default(), sess.graph.as_default():
            env = config.make_vec_env(renderer=True)
            obs = env.reset()
            render = False

            def toggle_rendering():
                print("Toggle rendering")
                nonlocal render
                render = not render

            while True:
                kp.keyboardCommands("r", toggle_rendering)
                if render:
                    env.render()
                    action, neglogp_action = estimator.get_action(obs)

                    obs, reward, done, info = env.step(action)
                    time.sleep(0.02)
                else:
                    time.sleep(1)

    _thread.start_new_thread(renderer_thread, (estimator, sess))

    runner = EnvRunner(sess, venv, estimator)
    # runner = SampleRunner(runner,sample_rate=10)
    for t in range(parameters.nb_updates):

        decay = t / parameters.nb_updates if parameters.decay else 0
        learning_rate = parameters.learning_rate * (1 - decay)
        clipping = parameters.clipping * (1 - decay)

        training_batch, epinfos = runner.run_timesteps(parameters.nb_steps)

        inds = np.arange(parameters.batch_size)
        for _ in range(parameters.nb_epochs):

            np.random.shuffle(inds)
            for start in range(0, parameters.batch_size, parameters.minibatch_size):
                end = start + parameters.minibatch_size
                mb_inds = inds[start:end]

                train_results = estimator.train(
                    obs=training_batch['obs'][mb_inds],
                    next_obs=training_batch['next_obs'][mb_inds],
                    values=training_batch['values'][mb_inds],
                    actions=training_batch['actions'][mb_inds],
                    neglogp_actions=training_batch['neglogp_actions'][mb_inds],
                    advantages=training_batch['advantages'][mb_inds],
                    returns=training_batch['returns'][mb_inds],
                    clipping=clipping,
                    learning_rate=learning_rate,
                )

                for train_result in train_results:
                    tboard.add(train_result, train_results[train_result])

        if t % (10) == 9:
            print("Saved model", t)
            saver.save(sess, save_path)

        if t % (1) == 0:
            print("Summary ", t * parameters.batch_size)
            tboard.save(t * parameters.batch_size)


if __name__ == "__main__":
    simulate()
