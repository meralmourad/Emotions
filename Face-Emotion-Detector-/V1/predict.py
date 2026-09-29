from .modeling.forward import forward_propagation
import numpy as np
from loguru import logger
import typer


def predict(X, Y, parameters):
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1)

    layer_count = len(parameters) // 2
    class_count = parameters['W' + str(layer_count)].shape[0]
    probabilities, _ = forward_propagation(X.T, class_count, parameters)
    predictions = np.argmax(probabilities, axis=0)

    if predictions.shape != Y.shape:
        raise ValueError("The number of input samples and labels must match.")

    accuracy = np.mean(predictions == Y)
    logger.info(f"Prediction accuracy: {accuracy:.2%}")
    return predictions