import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from networks.tf_networks import *
from utils.utils import *
import traces
from traces import tf_traces


class DoubleQLearning(tf.Module):
    def __init__(self, args, state_dims, num_actions):
        super(DoubleQLearning, self).__init__()
        self.args = args
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions)
        self.target_actor = ActorNetwork(args, state_dims, num_actions)
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
            vals = self.actor(states)
            vals = tf.gather_nd(params=vals, indices=tf.transpose(actions), batch_dims=1)
            next_vals = tf.expand_dims(tf.argmax(self.actor(next_states), axis=1),0)
            next_q_vals = self.target_actor(next_states)
            next_q_values = tf.gather_nd(params=next_q_vals, indices=tf.transpose(next_vals), batch_dims=1)
            target = tf.stop_gradient(rewards + self.args.gamma*next_q_values*(1 - dones))
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
        self.update_target(step_count)
        return to_np(self.args, td_error)[0]

    @tf.function
    def update_target(self, step_count):
        def update_ops(target_variable, source_variable):
            return target_variable.assign(source_variable, True)

        if step_count==1:
            update_vars = [update_ops(target_var, source_var) for target_var, source_var
                             in zip(self.target_actor.trainable_variables, self.actor.trainable_variables)]
            return tf.group(name="update_all_variables", *update_vars)

    def update_trace(self, actions):
        return getattr(tf_traces, self.args.trace)(self.args, actions, self.trace)

    @tf.function
    def reset_trace(self):
            return tf.zeros((self.args.batch_size, self.num_actions))















