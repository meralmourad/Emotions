from pathlib import Path
import cv2
import numpy as np
import typer
from loguru import logger
from tqdm import tqdm

from V1.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def load_and_flatten_dataset(data_path: Path, img_size: tuple[int, int] = (64, 64)):
    
    data_path = Path(data_path)

    X_list = []
    y_list = []

    folders = sorted([f for f in data_path.iterdir() if f.is_dir()])

    if not folders:
        logger.warning(f"No subdirectories found in: {data_path}")
        return np.array([]), np.array([]), {}

    label_map = {folder.name: i for i, folder in enumerate(folders)}
    logger.info(f"Label Map: {label_map}")

    for folder in tqdm(folders, desc="Processing categories"):
        label = label_map[folder.name]

        # Case-insensitive image filtering
        image_paths = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        for path in tqdm(image_paths, desc=f"Loading {folder.name}", leave=False):
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"Failed to read image: {path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, img_size)
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_flat = img_normalized.flatten()

            X_list.append(img_flat)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y, label_map


@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR / "train",
    output_dir: Path = PROCESSED_DATA_DIR,
    image_size: int = 64,
):
    """CLI entry point for dataset preprocessing and array export."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing raw dataset from: {input_dir}")
    X_train, y_train, label_map = load_and_flatten_dataset(
        data_path=input_dir, img_size=(image_size, image_size)
    )

    logger.info(f"Features matrix X shape: {X_train.shape}")
    logger.info(f"Target labels y shape: {y_train.shape}")

    np.save(output_dir / "train_X.npy", X_train)
    np.save(output_dir / "train_y.npy", y_train)

    logger.success(f"Saved processed arrays successfully to {output_dir}")


if __name__ == "__main__":
    app()