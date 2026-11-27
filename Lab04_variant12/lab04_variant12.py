from __future__ import annotations

import argparse
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path(__file__).with_name(".matplotlib")
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Perceptron
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings


DATASET_URLS = (
    "https://archive.ics.uci.edu/static/public/313/semg+for+basic+hand+movements.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00313/semg+for+basic+hand+movements.zip",
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
RAW_ZIP = DATA_DIR / "semg_basic_hand_movements.zip"

MOVEMENTS = {
    "spher": "Spherical",
    "tip": "Tip",
    "palm": "Palmar",
    "lat": "Lateral",
    "cyl": "Cylindrical",
    "hook": "Hook",
}


@dataclass(frozen=True)
class Split:
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if list(DATA_DIR.rglob("*.mat")):
        return

    last_error: Exception | None = None
    for url in DATASET_URLS:
        try:
            print(f"Downloading dataset: {url}")
            with urllib.request.urlopen(url, timeout=120) as response:
                with RAW_ZIP.open("wb") as file:
                    shutil.copyfileobj(response, file)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            if RAW_ZIP.exists():
                RAW_ZIP.unlink()
    else:
        raise RuntimeError(
            "Could not download the dataset. Download it manually from UCI and "
            f"place the extracted .mat files into {DATA_DIR.resolve()}."
        ) from last_error

    with zipfile.ZipFile(RAW_ZIP) as archive:
        archive.extractall(DATA_DIR)


def select_mat_files(database: str) -> list[Path]:
    mat_files = sorted(DATA_DIR.rglob("*.mat"))
    if database == "all":
        return mat_files

    wanted = "database 1" if database == "database1" else "database 2"
    return [path for path in mat_files if wanted in path.as_posix().lower()]


def zero_crossing_rate(signal: np.ndarray) -> float:
    centered = signal - np.mean(signal)
    return float(np.mean(centered[:-1] * centered[1:] < 0.0))


def slope_sign_change_rate(signal: np.ndarray) -> float:
    diff = np.diff(signal)
    if len(diff) < 2:
        return 0.0
    return float(np.mean(diff[:-1] * diff[1:] < 0.0))


def channel_features(signal: np.ndarray) -> list[float]:
    signal = np.asarray(signal, dtype=float).ravel()
    diff = np.diff(signal)
    return [
        float(np.mean(signal)),
        float(np.std(signal)),
        float(np.min(signal)),
        float(np.max(signal)),
        float(np.median(signal)),
        float(np.percentile(signal, 25)),
        float(np.percentile(signal, 75)),
        float(np.sqrt(np.mean(signal**2))),
        float(np.mean(np.abs(signal))),
        float(np.sum(np.abs(diff))),
        float(np.mean(diff**2)) if len(diff) else 0.0,
        zero_crossing_rate(signal),
        slope_sign_change_rate(signal),
    ]


def pair_features(channel_1: np.ndarray, channel_2: np.ndarray) -> list[float]:
    channel_1 = np.asarray(channel_1, dtype=float).ravel()
    channel_2 = np.asarray(channel_2, dtype=float).ravel()
    features = channel_features(channel_1) + channel_features(channel_2)

    min_length = min(len(channel_1), len(channel_2))
    if min_length > 1:
        correlation = float(np.corrcoef(channel_1[:min_length], channel_2[:min_length])[0, 1])
    else:
        correlation = 0.0
    if np.isnan(correlation):
        correlation = 0.0

    rms_1 = float(np.sqrt(np.mean(channel_1**2)))
    rms_2 = float(np.sqrt(np.mean(channel_2**2)))
    features.extend(
        [
            correlation,
            abs(rms_1 - rms_2),
            rms_1 / (rms_2 + 1e-12),
        ]
    )
    return features


def load_semg_dataset(database: str) -> pd.DataFrame:
    download_dataset()
    mat_files = select_mat_files(database)
    if not mat_files:
        raise RuntimeError(f"No .mat files found for database={database!r} in {DATA_DIR.resolve()}.")

    rows: list[dict[str, float | str]] = []
    for mat_file in mat_files:
        mat = loadmat(mat_file)
        database_name = "Database 1" if "database 1" in mat_file.as_posix().lower() else "Database 2"
        subject = mat_file.stem

        for key, label in MOVEMENTS.items():
            channel_1_name = f"{key}_ch1"
            channel_2_name = f"{key}_ch2"
            if channel_1_name not in mat or channel_2_name not in mat:
                continue

            channel_1 = np.asarray(mat[channel_1_name], dtype=float)
            channel_2 = np.asarray(mat[channel_2_name], dtype=float)
            trial_count = min(channel_1.shape[0], channel_2.shape[0])

            for trial_index in range(trial_count):
                feature_values = pair_features(channel_1[trial_index], channel_2[trial_index])
                row = {
                    f"feature_{feature_index:02d}": value
                    for feature_index, value in enumerate(feature_values, start=1)
                }
                row["label"] = label
                row["database"] = database_name
                row["subject"] = subject
                rows.append(row)

    data = pd.DataFrame(rows)
    feature_columns = [column for column in data.columns if column.startswith("feature_")]
    data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=feature_columns + ["label"]).reset_index(drop=True)
    return data


def manual_stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    train_size: float,
    validation_size: float,
    seed: int,
) -> Split:
    if not 0.0 < train_size < 1.0:
        raise ValueError("train_size must be between 0 and 1.")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    if train_size + validation_size >= 1.0:
        raise ValueError("train_size + validation_size must be less than 1.")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    for label in np.unique(y):
        label_indices = np.flatnonzero(y == label)
        label_indices = rng.permutation(label_indices)

        train_count = int(round(len(label_indices) * train_size))
        validation_count = int(round(len(label_indices) * validation_size))

        train_indices.extend(label_indices[:train_count])
        validation_indices.extend(label_indices[train_count : train_count + validation_count])
        test_indices.extend(label_indices[train_count + validation_count :])

    train_indices = rng.permutation(train_indices)
    validation_indices = rng.permutation(validation_indices)
    test_indices = rng.permutation(test_indices)

    return Split(
        x_train=x[train_indices],
        x_validation=x[validation_indices],
        x_test=x[test_indices],
        y_train=y[train_indices],
        y_validation=y[validation_indices],
        y_test=y[test_indices],
    )


def scale_split(split: Split) -> Split:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(split.x_train)
    x_validation = scaler.transform(split.x_validation)
    x_test = scaler.transform(split.x_test)
    return Split(
        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,
        y_train=split.y_train,
        y_validation=split.y_validation,
        y_test=split.y_test,
    )


def evaluate_classifier(
    name: str,
    model: Perceptron | MLPClassifier,
    split: Split,
) -> dict[str, float | str]:
    return {
        "model": name,
        "train_accuracy": accuracy_score(split.y_train, model.predict(split.x_train)),
        "validation_accuracy": accuracy_score(
            split.y_validation, model.predict(split.x_validation)
        ),
        "test_accuracy": accuracy_score(split.y_test, model.predict(split.x_test)),
    }


def run_perceptron_experiments(split: Split, seed: int) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for eta0 in [0.0005, 0.001, 0.01, 0.1, 1.0]:
        for alpha in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            model = Perceptron(
                eta0=eta0,
                alpha=alpha,
                penalty="l2",
                max_iter=2000,
                tol=1e-4,
                random_state=seed,
            )
            model.fit(split.x_train, split.y_train)
            metrics = evaluate_classifier("Perceptron", model, split)
            rows.append(
                {
                    **metrics,
                    "eta0": eta0,
                    "alpha": alpha,
                    "solver": "sgd",
                }
            )
            print(
                f"Perceptron eta0={eta0:g}, alpha={alpha:g}: "
                f"validation={metrics['validation_accuracy']:.4f}, "
                f"test={metrics['test_accuracy']:.4f}"
            )
    return pd.DataFrame(rows)


def run_mlp_experiments(split: Split, seed: int) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for solver in ["adam", "sgd", "lbfgs"]:
        for learning_rate_init in [0.0005, 0.001, 0.01]:
            for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:
                model = MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    solver=solver,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=500,
                    random_state=seed,
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(split.x_train, split.y_train)

                metrics = evaluate_classifier("MLPClassifier", model, split)
                rows.append(
                    {
                        **metrics,
                        "eta0": learning_rate_init,
                        "alpha": alpha,
                        "solver": solver,
                    }
                )
                print(
                    f"MLP solver={solver}, lr={learning_rate_init:g}, alpha={alpha:g}: "
                    f"validation={metrics['validation_accuracy']:.4f}, "
                    f"test={metrics['test_accuracy']:.4f}"
                )
    return pd.DataFrame(rows)


def plot_parameter_curves(
    results: pd.DataFrame,
    parameter: str,
    title: str,
    output_name: str,
    log_x: bool = False,
) -> None:
    grouped = (
        results.groupby(parameter, as_index=False)["validation_accuracy"]
        .max()
        .sort_values(parameter)
    )

    plt.figure(figsize=(8, 5))
    if log_x:
        plt.semilogx(grouped[parameter], grouped["validation_accuracy"], marker="o")
    else:
        plt.plot(grouped[parameter].astype(str), grouped["validation_accuracy"], marker="o")
    plt.xlabel(parameter)
    plt.ylabel("Best validation accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=160)
    plt.close()


def plot_solver_bar(results: pd.DataFrame, output_name: str) -> None:
    grouped = (
        results.groupby("solver", as_index=False)["validation_accuracy"]
        .max()
        .sort_values("validation_accuracy", ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(grouped["solver"], grouped["validation_accuracy"])
    plt.xlabel("solver")
    plt.ylabel("Best validation accuracy")
    plt.title("MLP solver comparison")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=160)
    plt.close()


def plot_final_accuracy(
    perceptron_best: pd.Series,
    mlp_best: pd.Series,
) -> None:
    labels = ["Train", "Validation", "Test"]
    perceptron_values = [
        perceptron_best["train_accuracy"],
        perceptron_best["validation_accuracy"],
        perceptron_best["test_accuracy"],
    ]
    mlp_values = [
        mlp_best["train_accuracy"],
        mlp_best["validation_accuracy"],
        mlp_best["test_accuracy"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, perceptron_values, width, label="Perceptron")
    plt.bar(x + width / 2, mlp_values, width, label="MLPClassifier")
    plt.xticks(x, labels)
    plt.ylabel("Accuracy")
    plt.title("Best models accuracy")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_models_accuracy.png", dpi=160)
    plt.close()


def plot_confusion_matrix(
    model: MLPClassifier,
    split: Split,
    label_names: list[str],
) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        split.y_test,
        model.predict(split.x_test),
        display_labels=label_names,
        xticks_rotation=35,
        cmap="Blues",
    )
    display.figure_.set_size_inches(8, 6)
    display.figure_.tight_layout()
    display.figure_.savefig(OUTPUT_DIR / "mlp_confusion_matrix.png", dpi=160)
    plt.close(display.figure_)


def train_mlp_from_row(row: pd.Series, split: Split, seed: int) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver=str(row["solver"]),
        alpha=float(row["alpha"]),
        learning_rate_init=float(row["eta0"]),
        max_iter=500,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(split.x_train, split.y_train)
    return model


def save_report(
    data: pd.DataFrame,
    split: Split,
    label_encoder: LabelEncoder,
    perceptron_best: pd.Series,
    mlp_best: pd.Series,
    classification_text: str,
    database: str,
) -> None:
    class_counts = data["label"].value_counts().sort_index()
    report = f"""# Результаты лабораторной работы №4, вариант 12

Датасет: sEMG for Basic Hand movements
Источник: https://archive.ics.uci.edu/dataset/313/semg+for+basic+hand+movements
Использованная часть датасета: `{database}`
Количество объектов: {len(data)}
Количество признаков после извлечения: {len([column for column in data.columns if column.startswith("feature_")])}
Классы: {", ".join(label_encoder.classes_)}

## Разделение выборки

- Train: {len(split.y_train)}
- Validation: {len(split.y_validation)}
- Test: {len(split.y_test)}

## Распределение классов

{class_counts.to_string()}

## Лучшая модель Perceptron

- eta0: {perceptron_best["eta0"]}
- alpha: {perceptron_best["alpha"]}
- Train accuracy: {perceptron_best["train_accuracy"]:.6f}
- Validation accuracy: {perceptron_best["validation_accuracy"]:.6f}
- Test accuracy: {perceptron_best["test_accuracy"]:.6f}

## Лучшая модель MLPClassifier

- solver: {mlp_best["solver"]}
- learning_rate_init: {mlp_best["eta0"]}
- alpha: {mlp_best["alpha"]}
- Train accuracy: {mlp_best["train_accuracy"]:.6f}
- Validation accuracy: {mlp_best["validation_accuracy"]:.6f}
- Test accuracy: {mlp_best["test_accuracy"]:.6f}

## Classification report для лучшего MLPClassifier на test

```text
{classification_text}
```

## Выходные файлы

- `processed_features.csv`
- `perceptron_results.csv`
- `mlp_results.csv`
- `best_model_metrics.csv`
- `perceptron_eta0_accuracy.png`
- `perceptron_alpha_accuracy.png`
- `mlp_learning_rate_accuracy.png`
- `mlp_alpha_accuracy.png`
- `mlp_solver_accuracy.png`
- `best_models_accuracy.png`
- `mlp_confusion_matrix.png`
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab04 variant 12: neural network classifiers for sEMG hand movements."
    )
    parser.add_argument(
        "--database",
        choices=["all", "database1", "database2"],
        default="all",
        help="Which part of the UCI archive to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for manual stratified split and model initialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_semg_dataset(args.database)
    feature_columns = [column for column in data.columns if column.startswith("feature_")]
    label_encoder = LabelEncoder()

    x = data[feature_columns].to_numpy(dtype=float)
    y = label_encoder.fit_transform(data["label"])

    split = manual_stratified_split(
        x,
        y,
        train_size=0.6,
        validation_size=0.2,
        seed=args.seed,
    )
    split = scale_split(split)

    print(f"Rows: {len(data)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Train / validation / test: {len(split.y_train)} / {len(split.y_validation)} / {len(split.y_test)}")

    perceptron_results = run_perceptron_experiments(split, args.seed)
    mlp_results = run_mlp_experiments(split, args.seed)

    perceptron_best = perceptron_results.sort_values(
        ["validation_accuracy", "test_accuracy"],
        ascending=False,
    ).iloc[0]
    mlp_best = mlp_results.sort_values(
        ["validation_accuracy", "test_accuracy"],
        ascending=False,
    ).iloc[0]

    best_mlp_model = train_mlp_from_row(mlp_best, split, args.seed)
    classification_text = classification_report(
        split.y_test,
        best_mlp_model.predict(split.x_test),
        target_names=list(label_encoder.classes_),
    )

    data.to_csv(OUTPUT_DIR / "processed_features.csv", index=False)
    perceptron_results.to_csv(OUTPUT_DIR / "perceptron_results.csv", index=False)
    mlp_results.to_csv(OUTPUT_DIR / "mlp_results.csv", index=False)
    pd.DataFrame([perceptron_best.to_dict(), mlp_best.to_dict()]).to_csv(
        OUTPUT_DIR / "best_model_metrics.csv",
        index=False,
    )

    plot_parameter_curves(
        perceptron_results,
        parameter="eta0",
        title="Perceptron learning rate vs validation accuracy",
        output_name="perceptron_eta0_accuracy.png",
        log_x=True,
    )
    plot_parameter_curves(
        perceptron_results,
        parameter="alpha",
        title="Perceptron regularization vs validation accuracy",
        output_name="perceptron_alpha_accuracy.png",
        log_x=True,
    )
    plot_parameter_curves(
        mlp_results,
        parameter="eta0",
        title="MLP learning rate vs validation accuracy",
        output_name="mlp_learning_rate_accuracy.png",
        log_x=True,
    )
    plot_parameter_curves(
        mlp_results,
        parameter="alpha",
        title="MLP regularization vs validation accuracy",
        output_name="mlp_alpha_accuracy.png",
        log_x=True,
    )
    plot_solver_bar(mlp_results, "mlp_solver_accuracy.png")
    plot_final_accuracy(perceptron_best, mlp_best)
    plot_confusion_matrix(best_mlp_model, split, list(label_encoder.classes_))

    save_report(
        data=data,
        split=split,
        label_encoder=label_encoder,
        perceptron_best=perceptron_best,
        mlp_best=mlp_best,
        classification_text=classification_text,
        database=args.database,
    )

    print("Best Perceptron:")
    print(perceptron_best.to_string())
    print("Best MLPClassifier:")
    print(mlp_best.to_string())
    print(f"Saved results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
