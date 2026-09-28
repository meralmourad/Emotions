import numpy as np 

def calc_cost(AL, C, Y):
    A = -np.mean(np.sum(Y * np.log(AL), axis=1))
    return A


def calc_cost_backward(AL, C, Y):
    m = AL.shape[0]
    dAL = -(Y / AL) / m
    return dAL