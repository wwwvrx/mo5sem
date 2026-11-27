from __future__ import annotations

import argparse
import json
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from keras import layers, models, utils
from keras.datasets import fashion_mnist
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


OUTPUT_DIR = Path("outputs")
MODEL_PATH = OUTPUT_DIR / "fashion_mnist_cnn.keras"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"

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


def prepare_images(images: np.ndarray) -> np.ndarray:
    images = images.astype("float32") / 255.0
    return np.expand_dims(images, axis=-1)


def load_data(train_limit: int, test_limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    if train_limit > 0:
        x_train = x_train[:train_limit]
        y_train = y_train[:train_limit]
    if test_limit > 0:
        x_test = x_test[:test_limit]
        y_test = y_test[:test_limit]

    return prepare_images(x_train), y_train, prepare_images(x_test), y_test


def build_model() -> models.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history_plot(history: dict[str, list[float]]) -> None:
    epochs = np.arange(1, len(history["loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["accuracy"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN training accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=160)
    plt.close()


def save_confusion_matrix(model: models.Model, x_test: np.ndarray, y_test: np.ndarray) -> str:
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=CLASS_NAMES,
        xticks_rotation=45,
        cmap="Blues",
    )
    display.figure_.set_size_inches(10, 8)
    display.figure_.tight_layout()
    display.figure_.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)

    return classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )


def save_sample_images(x_test: np.ndarray, y_test: np.ndarray, model: models.Model) -> None:
    sample_dir = OUTPUT_DIR / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict(x_test[:12], verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    for index, image in enumerate(x_test[:12]):
        plt.figure(figsize=(2, 2))
        plt.imshow(image.squeeze(), cmap="gray")
        plt.axis("off")
        title = f"true={CLASS_NAMES[y_test[index]]}, pred={CLASS_NAMES[predictions[index]]}"
        plt.title(title, fontsize=7)
        plt.tight_layout()
        plt.savefig(sample_dir / f"sample_{index:02d}.png", dpi=160)
        plt.close()


def save_report(
    args: argparse.Namespace,
    test_loss: float,
    test_accuracy: float,
    classification_text: str,
) -> None:
    report = f"""# Результаты лабораторной работы №6

Датасет: Fashion-MNIST
Библиотека: Keras 3
Backend: torch
Размер обучающей подвыборки: {args.train_limit}
Размер тестовой подвыборки: {args.test_limit}
Количество эпох: {args.epochs}
Batch size: {args.batch_size}

## Архитектура CNN

1. Conv2D(32, 3x3, ReLU)
2. MaxPooling2D(2x2)
3. Conv2D(64, 3x3, ReLU)
4. MaxPooling2D(2x2)
5. Dropout(0.25)
6. Flatten
7. Dense(128, ReLU)
8. Dropout(0.5)
9. Dense(10, Softmax)

## Итоговая точность

- Test loss: {test_loss:.6f}
- Test accuracy: {test_accuracy:.6f}

## Classification report

```text
{classification_text}
```

## Выходные файлы

- `fashion_mnist_cnn.keras`
- `training_history.json`
- `training_accuracy.png`
- `training_loss.png`
- `confusion_matrix.png`
- `sample_images/`
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Keras CNN on Fashion-MNIST.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--test-limit", type=int, default=3000)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    utils.set_random_seed(args.seed)

    x_train, y_train, x_test, y_test = load_data(args.train_limit, args.test_limit)
    y_train_categorical = utils.to_categorical(y_train, len(CLASS_NAMES))
    y_test_categorical = utils.to_categorical(y_test, len(CLASS_NAMES))

    model = build_model()
    model.summary()

    history = model.fit(
        x_train,
        y_train_categorical,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        verbose=2,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
    model.save(MODEL_PATH)

    history_data = {key: [float(value) for value in values] for key, values in history.history.items()}
    HISTORY_PATH.write_text(json.dumps(history_data, indent=2), encoding="utf-8")

    save_history_plot(history_data)
    classification_text = save_confusion_matrix(model, x_test, y_test)
    save_sample_images(x_test, y_test, model)
    save_report(args, float(test_loss), float(test_accuracy), classification_text)

    print(f"Test loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_accuracy:.6f}")
    print(f"Saved model to {MODEL_PATH.resolve()}")
    print(f"Saved report to {(OUTPUT_DIR / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
