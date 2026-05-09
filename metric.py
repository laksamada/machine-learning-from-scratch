import numpy as np

def mae(y_pred,target):
    return np.mean(np.abs(y_pred-target))

def mse(y_pred,target):
    return np.mean(np.square(y_pred-target))