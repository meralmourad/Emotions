import numpy as np

def softmax(Z, C):
    cache = Z = Z - np.max(Z , axis = 1 , keepdims = True)
    A = Z / np.sum(Z , axis= 1 , keepdims = True)

    return A , cache

def relu(Z):
    A = np.maximum(0 , Z)
    assert(A.shape == Z.shape)

    cache = Z

    return A , cache