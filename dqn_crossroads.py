import tensorflow as tf
from chi.rl.async import AsyncDQNAgent
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import experiment, model, Experiment

#from chi.rl.dqn import DqnAgent
from chi.rl.util import print_env, PenalizeAction, DiscretizeActions
from chi.util import log_top
#from chi.util import log_nvidia_smi

@experiment
def dqn_crossroad(self: Experiment, logdir=None, memory_size=100000, agents=1, replay_start=30000):
    from tensorflow.contrib import layers
    import gym
    from gym import spaces
    from gym import wrappers
    import numpy as np
    from tensorflow.contrib.framework import arg_scope
    from chi.rl import ReplayMemory

    chi.set_loglevel('debug')
    log_top(logdir + '/logs/top')
    #log_nvidia_smi(logdir + '/logs/nvidia-smi')

    memory = chi.rl.ReplayMemory(memory_size, 32)#?

    import argos3

    def make_env(i):
        env = gym.make('Crossroads-v0')
        env.unwrapped.conf(loglevel='info', logfile=logdir + f'/logs/unity_{i}')
        env = DiscretizeActions(env, [-1, 1])#2 actions: slow down, go faster (by a certain number)
        env = wrappers.Monitor(env, logdir + '/monitor_' + str(i),
                video_callable=lambda j: j % (50 if i==0 else 200) == 0)

        return env

    envs = [make_env(i) for i in range(agents)]
    monitor = envs[0]

    print_env(envs[0])

    @chi.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),
            optimizer=tf.train.RMSPropOptimizer(.00025, .95, .95, .01))
    def q_network(x):
        # might still miss one step or two for NN creation
        x /= 255
        # fully connected NN with 2 layers, each with 300 units
        x = layers.fully_connected(x, 300)
        x = layers.fully_connected(x, 300)
        return x

    agent = AsyncDQNAgent(envs, q_network, memory, replay_start=replay_start, logdir=logdir)
