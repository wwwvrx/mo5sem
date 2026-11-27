from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

DATA_DIR = Path(__file__).with_name("data")
KERAS_HOME = DATA_DIR / "keras"
KERAS_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(KERAS_HOME))

MPLCONFIGDIR = Path(__file__).with_name(".matplotlib")
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import numpy as np
from keras import models
from keras.datasets import fashion_mnist
from PIL import Image, ImageOps


MODEL_PATH = Path("outputs") / "fashion_mnist_cnn.keras"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def preprocess_array(image: np.ndarray) -> np.ndarray:
    image = image.astype("float32") / 255.0
    return image.reshape(1, 28, 28, 1)


def load_image_file(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)

    array = np.asarray(image)
    if np.mean(array) > 127:
        array = 255 - array
    return preprocess_array(array)


def load_test_sample(index: int) -> tuple[np.ndarray, int]:
    (_, _), (x_test, y_test) = fashion_mnist.load_data()
    if not 0 <= index < len(x_test):
        raise ValueError(f"sample-index must be between 0 and {len(x_test) - 1}.")
    return preprocess_array(x_test[index]), int(y_test[index])


def print_predictions(probabilities: np.ndarray, top_k: int, expected_label: int | None) -> None:
    probabilities = probabilities.ravel()
    order = np.argsort(probabilities)[::-1][:top_k]

    if expected_label is not None:
        print(f"Expected class: {expected_label} ({CLASS_NAMES[expected_label]})")

    print("Predictions:")
    for rank, class_index in enumerate(order, start=1):
        print(
            f"{rank}. {class_index} ({CLASS_NAMES[class_index]}): "
            f"{probabilities[class_index]:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a saved Keras CNN and classify an image or Fashion-MNIST test sample."
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--image", type=Path, help="Path to PNG/JPG image for classification.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Fashion-MNIST test sample index used when --image is not provided.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model file was not found: {args.model}. Run train_cnn.py first."
        )

    model = models.load_model(args.model)

    if args.image:
        image = load_image_file(args.image)
        expected_label = None
        print(f"Image: {args.image}")
    else:
        image, expected_label = load_test_sample(args.sample_index)
        print(f"Fashion-MNIST test sample index: {args.sample_index}")

    probabilities = model.predict(image, verbose=0)
    print_predictions(probabilities, args.top_k, expected_label)


if __name__ == "__main__":
    main()
