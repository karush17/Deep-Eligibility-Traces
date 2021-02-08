import os
import numpy as np
import tensorflow as tf

# one could also use the tf.tensor_scatter_nd_update function for updating the trace inplace. However, this requires a 2 dimensional action index tensor which might be difficult to maintina during training. 

def replacing(args, actions, trace):
    trace = tf.one_hot(tf.squeeze(tf.transpose(actions),1), trace.get_shape()[1])
    return trace

def accumulating(args, actions, trace):
    new_trace = tf.one_hot(tf.squeeze(tf.transpose(actions),1), trace.get_shape()[1])
    trace += new_trace
    return trace

def dutch(args, actions, trace):
    new_trace = tf.one_hot(tf.squeeze(tf.transpose(actions),1), trace.get_shape()[1])
    trace += (1-args.lr)*new_trace + new_trace
    return trace




