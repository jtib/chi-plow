import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.layers.utils import smart_cond
from tensorflow.python.ops.variable_scope import get_local_variable

import chi
from chi import experiment, model, Experiment
from chi.rl.async import AsyncDQNAgent

from chi.rl.util import print_env, PenalizeAction, DiscretizeActions
from chi.util import log_top

@chi.model(tracker=tf.train.ExponentialMovingAverage(1 - .0005),
        optimizer=tf.train.RMSPropOptimizer(.00025, .95, .95, .01))
def q_network(x):
    # fully connected NN with 2 layers, 300 and 600 units resp.
    x = layers.fully_connected(x, 300)
    x = layers.fully_connected(x, 600)
    return x
