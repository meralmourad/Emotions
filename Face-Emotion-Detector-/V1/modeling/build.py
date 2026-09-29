import time
import numpy as np
import h5py
from numpy.random import seed

from .backward import backward_propagation
from .forward import forward_propagation
from .loss import calc_cost
from .update import update_parames
from .initializing import initial_parameters


def build_model (X , Y , C , layer_dims , learning_rate , num_iterations , print_cost = True):

    np.random.seed(1)
    costs = []
    parameters =  initial_parameters(layer_dims)

    for i in range(num_iterations):
        AL , caches = forward_propagation(X , C , parameters)

        cost = calc_cost(AL, C, Y)

        grades = backward_propagation(AL , C , Y , caches)

        parameters = update_parames(parameters , grades , learning_rate)

        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

        if i % 100 == 0:
            costs.append(cost)

    return parameters , costs 



    