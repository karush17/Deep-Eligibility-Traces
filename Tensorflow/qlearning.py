import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from networks.tf_networks import *
from utils.utils import *
import traces



class QLearning(nn.Module):
    def __init__(self, args, state_dims, num_actions):
        super(QLearning, self).__init__()
        self.args = args
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions)
        self.value_net = ValueNetwork(args, state_dims, num_actions)
        self.opt_actor = optimizers.Adam(learning_rate=args.lr)
        self.opt_value = optimizers.Adam(learning_rate=args.lr)
        self.trace = tf.zeros((self.args.batch_size, self.num_actions))
        print(self.actor)
        print(self.opt_value)
    
    def forward(self, states):
        return self.actor(states)
    
    def get_actions(self, steps, states):
        return self.actor.get_actions(steps, states)
    
    def update(self, replay_buffer, steps, step_count):
        batch_size = self.args.batch_size
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = to_tensor(states)
        actions = tf.cast(to_tensor(actions), tf.int64)
        rewards = to_tensor(rewards)
        next_states = to_tensor(next_states)
        dones = to_tensor(dones)
        with tf.GradientTape() as tape:
            vals = self.actor(states)
            vals = tf.gather(params=vals, indices=tf.expand_dims(actions, axis=1), axis=1)
            vals = tf.squeeze(1)
            next_vals = self.actor(next_states).max(1)[0]
            target = tf.stop_gradient(rewards + self.args.gamma*next_vals*(1 - dones))
            td_error = (target - vals)**2
            # trace update
            if self.args.trace!='none':
                if step_count==0:
                    self.trace = self.reset_trace()
                self.trace  = self.update_trace(actions)
                td_error *= tf.gather(params=self.trace, indices=tf.expand_dims(actions, axis=1), axis=1)
                td_error = tf.reduce_mean(tf.squeeze(td_error, axis=1))
                self.trace *= self.args.gamma*self.args.lamb
            # td update
            grads = tape.gradient(td_error, self.actor.trainable_variables)
            self.opt_actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return td_error

    def update_trace(self, actions):
        return getattr(traces, self.args.trace)(self.args, actions, self.trace)

    def reset_trace(self):
            return tf.zeros((self.args.batch_size, self.num_actions))















