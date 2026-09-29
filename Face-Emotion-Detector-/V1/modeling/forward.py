import numpy as np
from copy import deepcopy
from .activation import softmax, relu

def linear_forward(W , b , A_prev):

    Z = np.dot(W , A_prev) + b

    cache = (A_prev , W , b)

    return Z , cache


def linear_activation_forward(W , b , A_prev , C , activation):

    Z , linear_cache = linear_forward(W , b , A_prev)

    if activation.strip().lower() == "softmax":
        A , activation_cache = softmax(Z , C)

    elif activation.strip().lower() == 'relu':
        A , activation_cache = relu(Z)

    cache = (linear_cache , activation_cache)

    return A , cache
    

def forward_propagation(X, C, parameters):
    
    A = X
    caches = [] 
    updated_parameters = deepcopy(parameters)
    L_layers = len(parameters) // 2

    for l in range(1 , L_layers):
        A_prev = A

        W = parameters['W' + str(l)] 
        b = parameters['b' + str(l)]

        A , cache = linear_activation_forward(W , b , A_prev , C ,'relu')
        caches.append(cache) # to calc dz , da 

    W = parameters['W' + str(L_layers)] 
    b = parameters['b' + str(L_layers)]
    AL , cache = linear_activation_forward(W , b , A , C ,'softmax')

    caches.append(cache)

    return AL , caches 





    
     