import numpy as np

from .activation import relu_backward , softmax_backward
from .loss import calc_cost_backward

def linear_backward(dZ , cache):
    A_prev , W , Z = cache 

    dW = np.dot(dZ , A_prev.T)
    db = np.sum(dZ , axis = 1 , keepdims = True)

    dA_prev = np.dot(W.T , dZ)

    return dA_prev , dW , db



def linear_activation_backward(dA, cache, activation):
    # dz           # da  
    linear_cache , activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA , activation_cache)

    if activation == 'softmax':
        dZ = softmax_backward(dA , activation_cache)

    dA_prev , dW , db = linear_backward(dZ , linear_cache)
    return dA_prev, dW, db


def backward_propagation(AL , C , Y , caches):

    grads = {}
    L = len(caches)

    dAL = calc_cost_backward(AL , C , Y)

    current_layer_cache = caches[L - 1]
    dA_prev, grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(
        dAL, current_layer_cache, 'softmax'
    )
    grads['dA' + str(L - 1)] = dA_prev


    for l in reversed(range(L - 1)):
        current_layer_cache = caches[l]
        dA_prev, grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = linear_activation_backward(
            dA_prev, current_layer_cache, 'relu'
        )
        grads['dA' + str(l)] = dA_prev

    return grads
