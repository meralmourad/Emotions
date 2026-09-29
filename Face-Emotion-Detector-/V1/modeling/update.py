from copy import deepcopy

def update_parames(parameters, grades, learning_rate):

    updated_parameters = deepcopy(parameters)
    layer_count = len(parameters) // 2

    for layer in range(1, layer_count + 1):
        updated_parameters['W' + str(layer)] -= grades['dW' + str(layer)] * learning_rate
        updated_parameters['b' + str(layer)] -= grades['db' + str(layer)] * learning_rate

    return updated_parameters