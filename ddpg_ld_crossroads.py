import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import Experiment
from chi.rl import ReplayMemory
from chi.rl.util import print_env
from chi.util import log_top
from chi.rl.ddpg import DdpgAgent
from chi.rl.wrappers import PenalizeAction

import gym
from gym import spaces, wrappers

import gym_plow7

import numpy as np

@chi.experiment
def ddpg_crossroads(self: Experiment, logdir=None):
    env = gym.make('plow7-v0')
    env = wrappers.Monitor(env, logdir + '/monitor', video_callable=None)
    env = PenalizeAction(env) #TODO: check if useful

    print_env(env)

    mem = ReplayMemory(100000) #TODO: check this value

    @chi.model(optimizer=tf.train.AdamOptimizer(.00005),
            tracker=tf.train.ExponentialMovingAverage(1-.0005))
    def preprocess(x):
        print(x.shape)
        x = tf.concat([tf.maximum(x, 0), -tf.minimum(x, 0)], 1)
        x = layers.fully_connected(x, 300)
        x = layers.fully_connected(x, 300)
        return x

    @chi.model(optimizer=tf.train.AdamOptimizer(0.0001),
            tracker = tf.train.ExponentialMovingAverage(1-.001))
    def actor(x, noise=False):
        x = layers.fully_connected(x, 300, biases_initializer=layers.xavier_initializer())
        x = layers.fully_connected(x, 300, biases_initializer=layers.xavier_initializer())
        a = layers.fully_connected(x, env.action_space.shape[0], None,
                weights_initializer=tf.random_normal_initializer(0, 1e-4))
        return a

    @chi.model(optimizer=tf.train.AdamOptimizer(.001),
            tracker=tf.train.ExponentialMovingAverage(1-.001))
    def critic(x, a):
        x = layers.fully_connected(x, 300,
                biases_initializer=layers.xavier_initializer())
        x = tf.concat([x, a], axis=1)
        x = layers.fully_connected(x, 300,
                biases_initializer=layers.xavier_initializer())
        x = layers.fully_connected(x, 300,
                biases_initializer=layers.xavier_initializer())
        q = layers.fully_connected(x, 1, None,
                weights_initializer=tf.random_normal_initializer(0, 1e-4))
        return tf.squeeze(q, 1)

    agent = DdpgAgent(env, actor, critic, preprocess, mem, training_repeats=5)

    for ep in range(100000):
        ret, _ = agent.play_episode()

        if ep % 100 == 0:
            print(f'Episode {ep}: R={ret}, t={agent.t})')
            getattr(getattr(env, 'unwrapped', env), 'report', lambda: None)()
