import numpy as np

def softmax(Z, C):
    shifted_Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(shifted_Z)
    A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    return A, A


def softmax_backward(dA, cache):
    A = cache
    dZ = A * (dA - np.sum(dA * A, axis=1, keepdims=True))

    assert dZ.shape == A.shape

    return dZ



def relu(Z):
    A = np.maximum(0 , Z)
    assert(A.shape == Z.shape)

    cache = Z

    return A , cache


def relu_backward(dA , cache):
    Z = cache
    dZ = np.array(dA, copy=True) 
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ