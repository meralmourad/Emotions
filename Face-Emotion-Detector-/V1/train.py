from pathlib import Path
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage  
from numpy.random import seed

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from loguru import logger
import typer

from .config import RAW_DATA_DIR
from .dataset import load_and_flatten_dataset
from .modeling.build import build_model
from .predict import predict


app = typer.Typer()


def stratified_train_test_split(X, Y, train_fraction=0.6, random_state=42):
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1)

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("The number of samples and labels must match.")

    labels, class_counts = np.unique(Y, return_counts=True)
    if np.any(class_counts < 2):
        raise ValueError("Each class needs at least two samples for a stratified split.")

    rng = np.random.default_rng(random_state)
    exact_train_counts = class_counts * train_fraction
    train_counts = np.floor(exact_train_counts).astype(int)
    remaining = round(len(Y) * train_fraction) - int(train_counts.sum())
    fractional_counts = exact_train_counts - train_counts
    order = np.lexsort((rng.random(len(labels)), -fractional_counts))
    train_counts[order[:remaining]] += 1

    train_indices = []
    test_indices = []
    for label, train_count in zip(labels, train_counts):
        indices = np.flatnonzero(Y == label)
        rng.shuffle(indices)
        train_indices.extend(indices[:train_count])
        test_indices.extend(indices[train_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)

    return X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]


@app.command()
def main(
    input_train_dir: Path = RAW_DATA_DIR / "train",
    train_fraction: float = 0.6,
):
    X_all, Y_all, train_label_map = load_and_flatten_dataset(input_train_dir)
    X_train, Y_train, X_test, Y_test = stratified_train_test_split(
        X_all, Y_all, train_fraction=train_fraction
    )

    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Training labels shape: {Y_train.shape}")
    logger.info(f"Test features shape: {X_test.shape}")
    logger.info(f"Test labels shape: {Y_test.shape}")
    logger.info(f"Number of classes: {len(train_label_map)}")
    # index = 10
    # image = Image.fromarray(
    #     (X[index].reshape(64, 64, 3) * 255).astype(np.uint8)
    # )
    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()
    # print("y = " + str(Y[index]))

    C = len(train_label_map)

    Y_one_hot = np.eye(C, dtype=np.float32)[Y_train.reshape(-1)].T
    layers_dims = [X_train.shape[1], 20, 7, 5, C]
    parameters, costs = build_model(
        X_train.T, Y_one_hot, C, layers_dims,
        learning_rate=0.075, num_iterations=2501, print_cost=True
    )
    predictions_test = predict(X_test, Y_test, parameters)

    cost_iterations = np.arange(len(costs)) * 100
    plt.figure()
    plt.plot(cost_iterations, costs, marker="o")
    plt.xlabel("Training iteration")
    plt.ylabel("Cost")
    plt.title("Training cost over time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




    


if __name__ == "__main__":
    app()
