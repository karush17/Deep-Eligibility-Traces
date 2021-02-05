import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from networks.tf_networks import *
from utils.utils import *

class WatkinsQ(tf.Module):
    def __init__(self, args, state_dims, num_actions):
        super(WatkinsQ, self).__init__()
        self.args = args
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions)
        self.opt_actor = optimizers.Adam(learning_rate=self.args.lr)
        self.trace = {}
        for idx, p in enumerate(self.actor.trainable_variables):
            self.trace[idx] = tf.zeros(p.get_shape())
        print(self.actor)
        print(self.opt_actor)
    
    def __call__(self, states):
        return self.actor(states)
    
    def get_actions(self, steps, states):
        return self.actor.get_actions(steps, states)

    @tf.function
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
            vals = tf.gather_nd(params=vals, indices=tf.transpose(actions), batch_dims=1)
            next_vals = tf.reduce_max(self.actor(next_states), axis=1)
            target = tf.stop_gradient(rewards + self.args.gamma*next_vals*(1 - dones))
            td_error = (target - vals)**2
            td_error = tf.reduce_mean(td_error, axis=1)
            self.trace = self.reset_trace(step_count) 
            grads = tape.gradient(td_error, self.actor.trainable_variables)
            new_grads = []
            for idx, p in enumerate(self.actor.trainable_variables):
                if idx not in list(self.trace.keys()):
                    self.trace = self.reset_trace(step_count, force_reset=True) 
                self.trace[idx] = self.args.gamma*self.args.lamb*self.trace[idx] + grads[idx]
                new_grads.append(td_error*self.trace[idx])
            grads = tuple(new_grads)
            self.opt_actor.apply_gradients(zip(grads, self.actor.trainable_variables))
        return to_np(self.args, td_error)[0]

    @tf.function
    def reset_trace(self, step_count, force_reset=False):
        if step_count==1 or force_reset:
            for idx, p in enumerate(self.actor.trainable_variables):
                self.trace[idx] = tf.zeros(p.get_shape())
        return self.trace
















