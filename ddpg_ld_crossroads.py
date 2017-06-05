import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import model
from chi.rl import ReplayMemory
from chi.rl.async import AsyncDQNAgent
from chi.rl.util import print_env
from chi.util import log_top
from chi.rl.ddpg import DdpgAgent
from chi.rl.wrappers import PenalizeAction

import gym
from gym import spaces, wrappers

import gym_plow7

import numpy as np

@experiment
def ddpg_crossroads(self: Experiment, logdir=None):
    env = gym.make('plow7-v0')
    env = wrappers.Monitor(env, logdir + '/monitor', video_callable=None)
    env = PenalizeAction(env) #TODO: check if useful

    print_env(env)

    m = ReplayMemory(100000) #TODO: check this value

    @chi.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),
            optimizer=tf.train.RMSPropOptimizer(.00025, .95, .95, .01))#is this the best optimizer for this case?
    def q_network(x):
        # fully connected NN with 2 layers, 300 and 600 units resp.
        x = layers.fully_connected(x, 300)
        x = layers.fully_connected(x, 600)
        return x
