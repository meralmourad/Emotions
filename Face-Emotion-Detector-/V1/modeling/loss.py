import numpy as np 

def calc_cost(AL, C, Y):
    A = -np.mean(np.sum(Y * np.log(AL), axis=1))
    return A