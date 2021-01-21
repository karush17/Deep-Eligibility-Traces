import os
import numpy as np
import torch
import tensorflow as tf

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_np(args.lib, x):
    if args.lib=='torch':
    	return x.detach().cpu().numpy()
    else:
        return tf.make_ndarray(x)

def to_torch(x, **kwargs):
	if torch.is_tensor(x):
		return x
	else:
		return torch.tensor(x, device=DEVICE, dtype=torch.float32, **kwargs)

def to_tensor(x):
    if tf.is_tensor(x):
        return x
    else:
        return tf.convert_to_tensor(x, dtype=tf.float32)

def dict_to_torch(d):
	return {
		key: to_torch(val)
		for key, val in d.items()
	}

def dict_to_tensor(d):
	return {
		key: to_tensor(val)
		for key, val in d.items()
	}

