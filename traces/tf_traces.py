import os
import numpy as np
import tensorflow as tf

def replacing(args, actions, trace):
    trace = tf.one_hot(tf.squeeze(tf.transpose(actions),1), trace.get_shape()[1])
    return trace

def accumulating(args, actions, trace):
    trace = trace.scatter_(1, actions.unsqueeze(1), 1, reduce='add')
    return trace

def dutch(args, actions, trace):
    trace = trace.scatter_(1, actions.unsqueeze(1), 1-args.lr, reduce='multiply')
    trace = trace.scatter_(1, actions.unsqueeze(1), 1, reduce='add')
    return trace




