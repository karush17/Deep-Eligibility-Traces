import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils.utils import *


class ActorNetwork(tf.Module):
    def __init__(self, num_actions):
        super(ActorNetwork, self).__init__()
        self.l1 = layers.Dense(128, activation='ReLU')
        self.l2 = layers.Dense(128, activation='ReLU')
        self.logits = layers.Dense(num_actions)

    def call(self, states):
        x = to_tensor(states)
        x = self.l1(x)
        x = self.l2(x)
        x = self.logits(x)
        return x
    
    def get_action(self, obs):
        x = self.__call__(obs)
        return np.squeeze(x, axis=-1)


class ValueNetwork(tf.Module):
    def __init__(self, num_actions):
        super(ValueNetwork, self).__init__(self)
        self.l1 = layers.Dense(128, activation='ReLU')
        self.l2 = layers.Dense(128, activation='ReLU')
        self.vals = layers.Dense(1)

    def call(self, states):
        x = to_tensor(states)
        x = self.l1(x)
        x = self.l2(x)
        x = self.vals(x)
        return x
    








