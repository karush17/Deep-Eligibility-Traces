import os
import csv
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_np(args, x):
    if args.lib=='torch':
    	return x.detach().cpu().numpy()
    else:
        return tf.make_ndarray(tf.stop_gradient(x))

def to_torch(x, **kwargs):
	if torch.is_tensor(x):
		return x.to(DEVICE)
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

def res_plot(args, log_dir):
	# plot rewards
    plt.figure()
    plt.title('Average Returns', fontsize=24)
    plt.plot(log_dir['ep_count'], log_dir['average_rewards'])
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('Returns', fontsize=18)
    plt.savefig('plot_rewards.png', dpi=600, bbox_inches='tight')
	# plot errors
    plt.figure()
    plt.title('Average TD Error', fontsize=24)
    plt.plot(log_dir['ep_count'], log_dir['average_error'])
    plt.xlabel('Steps', fontsize=18)
    plt.ylabel('Error', fontsize=18)
    plt.savefig(args.log_dir+'plot_error.png', dpi=600, bbox_inches='tight')

def save_logs(args, log_dir):
    with open(args.log_dir+args.alg+'_'+args.env+'_'+str(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))+'.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in log_dir.items():
            writer.writerow([key, value])

