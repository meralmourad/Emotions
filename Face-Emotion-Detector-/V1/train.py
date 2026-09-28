from pathlib import Path

from loguru import logger
import typer

from V1.config import RAW_DATA_DIR
from V1.dataset import load_and_flatten_dataset
from V1.modeling.backward import backward_propagation
from V1.modeling.forward import forward_propagation
from V1.modeling.loss import calc_cost
from V1.modeling.update import update_parames

app = typer.Typer()


@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR / "train",
):
    X_train, y_train, label_map = load_and_flatten_dataset(input_dir)

    logger.info(f"Training features shape: {X_train.shape}")
    logger.info(f"Training labels shape: {y_train.shape}")
    logger.info(f"Number of classes: {len(label_map)}")


if __name__ == "__main__":
    app()
