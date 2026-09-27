import numpy as np

def initial_parameters(layer_dims: list[int]):
    parameters = {}

    Layers = len(layer_dims)

    for l in range(1 , Layers):
        parameters['w' + str(l)] = np.random.randn(layer_dims[l] , layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l] , 1))

    assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
    assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters
        