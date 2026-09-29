import numpy as np 

def calc_cost(AL, C, Y):
    return -np.sum(Y * np.log(np.clip(AL, 1e-12, 1.0))) / Y.shape[1]


def calc_cost_backward(AL, C, Y):
    m = AL.shape[1]
    return -(Y / np.clip(AL, 1e-12, 1.0)) / m