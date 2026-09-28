import numpy as np

from .activation import relu_backward , softmax_backward
from .loss import calc_cost_backward

def linear_backward(dZ , cache):
    A_prev , W , Z = cache 
    m = A_prev.shape[1]

    dW = np.dot(dZ , A_prev.T) / m
    db = np.sum(dZ , axis = 1 , keepdims = True) / m 

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


def backward_propagation(AL , C , Y , caches):

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape(1))

    dAL = calc_cost_backward(AL , C , Y)

    current_layer_cache = caches[L - 1]
    grads['dA' + str(L - 1)] , grads['dW' + str(L)] , grads['db' + str(L)] = linear_activation_backward(dAL, current_layer_cache, 'softmax') 


    for l in reversed(range(L - 1)):
        current_layer_cache = caches[l]
        grads['dA' + str(l)] , 
        grads['dW' + str(l + 1)] ,
        grads['db' + str(l + 1)] = linear_activation_backward(dAL, current_layer_cache, 'softmax')

    return grads
