from numpy import copy

def update_parames(parames , grades , learrning_rate):

    parameters = copy.deepcopy(parames)
    L = parameters // 2 

    for l in range(L):
        parames['W' + str(l)] -= grades['dW' + str(l)] * learrning_rate
        parames['b' + str(l)] -= grades['db' + str(l)] * learrning_rate

    return parameters 