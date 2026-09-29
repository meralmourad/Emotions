import numpy as np

def initial_parameters(layer_dims):
    parameters = {}

    np.random.seed(3)

    Layers = len(layer_dims)

    for l in range(1 , Layers):
        if l == Layers - 1:
            scale = np.sqrt(1.0 / layer_dims[l - 1])
        else:
            scale = np.sqrt(2.0 / layer_dims[l - 1])

        parameters['W' + str(l)] = np.random.randn(layer_dims[l] , layer_dims[l - 1]) * scale
        parameters['b' + str(l)] = np.zeros((layer_dims[l] , 1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters
        