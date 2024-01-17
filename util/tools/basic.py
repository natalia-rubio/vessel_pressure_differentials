import sys
sys.path.append("/home/nrubio/Desktop/vessel_pressure_differentials")
import numpy as np
import pickle
import copy
import pdb
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
#plt.rc('text', usetex=True)
colors = ["royalblue", "orangered", "seagreen", "peru", "blueviolet"]

def dP_poiseuille(flow, radius, length):
    mu = 0.04
    dP = 8 * mu * length * flow /(np.pi * radius**4)
    return dP

def get_length(locs):
    length = 0
    for i in range(1, locs.shape[0]):
        length += np.linalg.norm(locs[i, :] - locs[i-1, :])
    return length

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_unif_random(stats_list, lower_rng_bound = 0.4, upper_rng_bound = 0.6):
    #stats_list = [min, max, avg, std]
    return stats_list[0] + (lower_rng_bound + \
                            (upper_rng_bound-lower_rng_bound) * np.random.default_rng(seed=0).random()) * \
                            (stats_list[1]-stats_list[0])

def get_middle(stats_list):
    return stats_list[0] + 0.5 * (stats_list[1]-stats_list[0])

# def get_percentile(stats_list, percentile):

#     return stats_list[0] + (stats_list[1]-stats_list[0])*0.4 + percentile * (stats_list[1]-stats_list[0])*0.2

def scale(scaling_dict, field, field_name):
    if scaling_dict[field_name][1] == 0:
        scaled_field = field
    else:
        mean = scaling_dict[field_name][0]; std = scaling_dict[field_name][1]
        scaled_field = (field-mean)/std
    return scaled_field

def inv_scale(scaling_dict, field, field_name):
    mean = scaling_dict[field_name][0]; std = scaling_dict[field_name][1]
    scaled_field = (field*std)+mean
    return scaled_field

def compute_mse(model_output, obs_output):
    """
    get mean squared error between two arrays
    """
    MSE = sum(np.square(
        model_output.flatten() - obs_output.flatten())
        ) / np.size(obs_output.flatten())  # compute MSE
    return MSE

def get_angle_diff(angle1, angle2):
    try:
        angle_diff = np.arccos(angle1.T @ angle2).reshape(-1,)
    except:
        pdb.set_trace()
    return angle_diff

def get_r2(x, y, coef):
    n = y.size
    y_mean = np.mean(y)
    if x.shape[1] == 2:
        y_pred = x[:,0] * coef[0] +  x[:,1]*coef[1]
    elif x.shape[1] == 3:
        y_pred = x[:,0] * coef[0] +  x[:,1]*coef[1] +  x[:,2]*coef[2]
    else:
        if coef.size == 1:
            y_pred = x * coef[0]
        if coef.size == 2:
            y_pred = x * coef[0] + coef[1]
    SST = np.sum(np.square(y.reshape(-1,) - y_mean.reshape(-1,)))
    SSE = np.sum(np.square(y.reshape(-1,) - y_pred.reshape(-1,)))

    r2 = 1 - SSE/SST
    return r2

def get_outlier_inds(data, m = 2):
    data_array = np.asarray(data)

    u = np.mean(data)
    s = np.std(data)

    outlier_inds = []; non_outlier_inds = []
    for i in range(int(len(data)/2)):

        i1_out =  u - m * s >  data[2*i] or data[2*i] > u + m * s
        i2_out =  u - m * s >  data[2*i+1] or data[2*i+1] > u + m * s

        if i1_out or i2_out:

            outlier_inds = outlier_inds + [2*i, 2*i + 1]
        else:
            non_outlier_inds = non_outlier_inds + [2*i, 2*i + 1]

    return outlier_inds, non_outlier_inds

def get_random_ind(num_pts, percent_train = 85 , seed = 0):

    ind = np.linspace(0,num_pts, num_pts, endpoint = False).astype(int); rng = np.random.default_rng(seed)
    train_ind = rng.choice(ind, size = int(num_pts * 0.01 * percent_train), replace = False)
    val_ind = ind[np.isin(ind, train_ind, invert = True)]
    return train_ind, val_ind
