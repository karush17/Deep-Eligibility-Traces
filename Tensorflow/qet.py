import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from networks.tf_networks import *
from utils.utils import *
from traces import tf_traces

class ExpectedTrace(tf.Module):
    def __init__(self, args, state_dims, num_actions):
        super(ExpectedTrace, self).__init__()
        self.args = args
        self.eta = 0.95
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.actor = ActorNetwork(args, state_dims, num_actions)
        self.exp_trace_param = tf.Variable(tf.random.uniform((self.num_actions, self.state_dims)), trainable=True)
        self.opt_actor = optimizers.Adam(learning_rate=self.args.lr)
        self.opt_trace = optimizers.Adam(learning_rate=self.args.lr)
        self.trace = {}
        for idx, p in enumerate(self.actor.trainable_variables):
            self.trace[idx] = tf.zeros(p.get_shape())

        print(self.actor)
        print(self.opt_actor)
    
    def forward(self, states):
        return self.actor(states)
    
    def get_actions(self, steps, states):
        return self.actor.get_actions(steps, states)

    def update(self, replay_buffer, steps, step_count):
        batch_size = self.args.batch_size
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = tf.stop_gradient(to_tensor(states))
        actions = tf.cast(to_tensor(actions), tf.int64)
        rewards = to_tensor(rewards)
        next_states = to_tensor(next_states)
        dones = to_tensor(dones)
        with tf.GradientTape() as trace_tape:
            trace_tape.watch(self.exp_trace_param)
            vals = self.actor(states)
            vals = tf.gather_nd(params=vals, indices=tf.transpose(actions), batch_dims=1)
            next_vals = tf.reduce_max(self.actor(next_states), axis=1)
            # expected trace update- The paper does not backpropagate trace gradients into feature representations. We follow this convention but reduce one dimension in the linear approximation since we take the expectation over batch.
            if list(self.trace.keys())==[]:
                    self.trace = self.reset_trace(step_count, force_reset=True) 
            ind_trace = list(self.trace.keys())[-1]
            self.exp_trace = tf.matmul(self.exp_trace_param, tf.transpose(states))
            trace_loss = (tf.reduce_mean(self.exp_trace,1) - self.trace[ind_trace])**2 # learn expectation from last layer
            trace_loss = tf.reduce_mean(trace_loss)
            trace_grad = trace_tape.gradient(trace_loss, self.exp_trace_param)[0]
        self.exp_trace_param = tf.map_fn(fn=lambda t: t+(self.args.lr*trace_grad), elems=self.exp_trace_param)
        # self.exp_trace_param.assign_add(self.args.lr*trace_grad)
        self.trace[ind_trace] = (1-self.eta)*tf.reduce_mean(self.exp_trace,1) + self.eta*self.trace[ind_trace]
        # TD update
        with tf.GradientTape() as tape:
            vals = self.actor(states)
            vals = tf.gather_nd(params=vals, indices=tf.transpose(actions), batch_dims=1)
            target = tf.stop_gradient(rewards + self.args.gamma*next_vals*(1 - dones))
            td_error = (target - vals)**2
            td_error = tf.reduce_mean(td_error, axis=1)
            self.trace = self.reset_trace(step_count) 
        eval_gradients = tape.gradient(td_error, self.actor.trainable_variables)
        # print(eval_gradients)
        new_grads = []
        for idx, p in enumerate(self.actor.trainable_variables):
            if idx not in list(self.trace.keys()):
                self.trace = self.reset_trace(step_count, force_reset=True) 
            self.trace[idx] = self.args.gamma*self.args.lamb*self.trace[idx] + eval_gradients[idx]
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
















