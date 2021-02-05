import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils.utils import *


class ActorNetwork(tf.Module):
    def __init__(self, args, num_inputs, num_actions):
        super(ActorNetwork, self).__init__()
        self.num_actions = num_actions
        self.args = args
        self.l1 = layers.Dense(128, activation='relu')
        self.l2 = layers.Dense(128, activation='relu')
        self.logits = layers.Dense(num_actions)

    def __call__(self, states):
        x = to_tensor(states)
        x = self.l1(x)
        x = self.l2(x)
        x = self.logits(x)
        return x
    
    def get_actions(self, steps, states):
        # select action during updates
        if states.shape[0]==self.args.batch_size:
            if random.random() > epsilon_by_step(self.args, steps):
                x = to_tensor(states)
                x = self.__call__(x)
                x = tf.cast(tf.argmax(x, axis=1), tf.int64)
                return tf.expand_dims(x, axis=0)
            else:
                return tf.cast(to_tensor(np.random.randint(low=0, high=self.num_actions, size=self.args.batch_size)), tf.int64)#random.randrange(self.num_actions)
            
        else:
            # select action during policy execution
            if random.random() > epsilon_by_step(self.args, steps):
                x = to_tensor(states)
                x = self.__call__(x)
                x = to_np(self.args, x)
                x = np.argmax(x, axis=1)[0]
                return x
            else:
                return random.randrange(self.num_actions)

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
    








