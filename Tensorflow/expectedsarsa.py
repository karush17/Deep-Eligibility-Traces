import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from networks.tf_networks import *
from utils.utils import *
import traces
from traces import tf_traces


class ExpectedSARSA(tf.Module):
    def __init__(self, args, state_dims, num_actions):
        super(ExpectedSARSA, self).__init__()
        self.args = args
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions)
        self.opt_actor = optimizers.Adam(learning_rate=self.args.lr)
        self.trace = tf.zeros((self.args.batch_size, self.num_actions))
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
            vals = self.actor(states)#[self.actor.get_actions(states)]
            vals = tf.gather_nd(params=vals, indices=tf.transpose(actions), batch_dims=1)
            next_vals = self.actor(next_states)
            next_vals = self.get_expectation(next_vals, epsilon_by_step(self.args, steps))
            target = tf.stop_gradient(rewards + self.args.gamma*next_vals*(1 - dones))
            td_error = (target - vals)**2
            # trace update
            if self.args.trace!='none':
                if step_count==0:
                    self.trace = self.reset_trace()
                self.trace  = self.update_trace(actions)
                td_error *= tf.gather_nd(params=self.trace, indices=tf.transpose(actions), batch_dims=1)
                self.trace *= self.args.gamma*self.args.lamb
            td_error = tf.reduce_mean(td_error, axis=1)
            # td update
        grads = tape.gradient(td_error, self.actor.trainable_variables)
        self.opt_actor.apply_gradients(zip(grads, self.actor.trainable_variables))
        return to_np(self.args, td_error)[0]

    @tf.function
    def get_expectation(self, q_vals, epsilon):
        # get indices of best actions
        idx = tf.argmax(q_vals, axis=1)
        idx = tf.transpose(tf.expand_dims(idx, axis=0))
        # gather best actions
        best_acts = tf.gather_nd(params=q_vals, indices=idx, batch_dims=1)
        # create mask to make best actions zero in the original tensor
        mask = tf.one_hot(tf.squeeze(idx,1), self.num_actions)
        # mask = tf.tensor_scatter_nd_update(tf.ones_like(q_vals), idx, tf.zeros(tf.shape(idx)))
        # get sub-optimal actions by applying mask
        sub_acts = q_vals*mask
        return (1-epsilon)*best_acts + (epsilon/(self.num_actions-1))*tf.reduce_sum(sub_acts, axis=1)


    def update_trace(self, actions):
        return getattr(tf_traces, self.args.trace)(self.args, actions, self.trace)

    @tf.function
    def reset_trace(self):
            return tf.zeros((self.args.batch_size, self.num_actions))
















