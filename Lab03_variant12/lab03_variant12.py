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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


DATASET_URLS = (
    "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip",
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
RAW_TXT = DATA_DIR / "household_power_consumption.txt"
RAW_ZIP = DATA_DIR / "household_power_consumption.zip"

FEATURE_COLUMNS = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
    "day_of_week",
    "month",
]
TARGET_COLUMN = "Global_active_power"


@dataclass(frozen=True)
class Metrics:
    model: str
    train_r2: float
    test_r2: float
    test_mse: float
    test_mae: float


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_TXT.exists():
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
            f"place household_power_consumption.txt into {DATA_DIR.resolve()}."
        ) from last_error

    with zipfile.ZipFile(RAW_ZIP) as archive:
        archive.extract("household_power_consumption.txt", DATA_DIR)


def load_dataset(max_rows: int) -> pd.DataFrame:
    download_dataset()

    data = pd.read_csv(
        RAW_TXT,
        sep=";",
        na_values="?",
        low_memory=False,
    )

    numeric_columns = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    timestamp = pd.to_datetime(
        data["Date"] + " " + data["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    data["hour"] = timestamp.dt.hour
    data["day_of_week"] = timestamp.dt.dayofweek
    data["month"] = timestamp.dt.month

    data = data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().reset_index(drop=True)

    if max_rows > 0 and len(data) > max_rows:
        row_index = np.linspace(0, len(data) - 1, max_rows, dtype=int)
        data = data.iloc[row_index].reset_index(drop=True)

    return data


def manual_train_test_split(
    features: np.ndarray,
    target: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    test_count = int(round(len(features) * test_size))

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    return (
        features[train_indices],
        features[test_indices],
        target[train_indices],
        target[test_indices],
    )


def evaluate_model(
    model_name: str,
    model: Pipeline | LinearRegression,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Metrics:
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    return Metrics(
        model=model_name,
        train_r2=r2_score(y_train, train_predictions),
        test_r2=r2_score(y_test, test_predictions),
        test_mse=mean_squared_error(y_test, test_predictions),
        test_mae=mean_absolute_error(y_test, test_predictions),
    )


def build_linear_model(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Metrics:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regression", LinearRegression()),
        ]
    )
    model.fit(x_train, y_train)
    return evaluate_model("LinearRegression", model, x_train, x_test, y_train, y_test)


def polynomial_experiments(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    degrees: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for degree in degrees:
        model = Pipeline(
            [
                ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scaler", StandardScaler()),
                ("regression", LinearRegression()),
            ]
        )
        model.fit(x_train, y_train)
        metrics = evaluate_model(
            f"Polynomial degree {degree}",
            model,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        rows.append({"degree": degree, **metrics.__dict__})
        print(
            f"Polynomial degree={degree}: "
            f"train R2={metrics.train_r2:.4f}, test R2={metrics.test_r2:.4f}"
        )
    return pd.DataFrame(rows)


def ridge_experiments(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alphas: np.ndarray,
    degree: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for alpha in alphas:
        model = Pipeline(
            [
                ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
                ("scaler", StandardScaler()),
                ("regression", Ridge(alpha=float(alpha))),
            ]
        )
        model.fit(x_train, y_train)
        metrics = evaluate_model(
            f"Ridge alpha {alpha:.5g}",
            model,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        rows.append({"alpha": float(alpha), "degree": degree, **metrics.__dict__})
        print(
            f"Ridge alpha={alpha:.5g}: "
            f"train R2={metrics.train_r2:.4f}, test R2={metrics.test_r2:.4f}"
        )
    return pd.DataFrame(rows)


def save_polynomial_plot(results: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(results["degree"], results["train_r2"], marker="o", label="Train R2")
    plt.plot(results["degree"], results["test_r2"], marker="o", label="Test R2")
    plt.xlabel("Polynomial degree")
    plt.ylabel("R2 score")
    plt.title("Polynomial degree vs model quality")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "polynomial_degree_r2.png", dpi=160)
    plt.close()


def save_ridge_plot(results: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    plt.semilogx(results["alpha"], results["train_r2"], marker="o", label="Train R2")
    plt.semilogx(results["alpha"], results["test_r2"], marker="o", label="Test R2")
    plt.xlabel("Ridge alpha")
    plt.ylabel("R2 score")
    plt.title("Regularization coefficient vs model quality")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ridge_alpha_r2.png", dpi=160)
    plt.close()


def save_report(
    linear_metrics: Metrics,
    polynomial_results: pd.DataFrame,
    ridge_results: pd.DataFrame,
    data_rows: int,
    max_rows: int,
) -> None:
    best_polynomial = polynomial_results.sort_values("test_r2", ascending=False).iloc[0]
    best_ridge = ridge_results.sort_values("test_r2", ascending=False).iloc[0]

    report = f"""# Результаты лабораторной работы №3, вариант 12

Датасет: Individual household electric power consumption
Целевая переменная: `{TARGET_COLUMN}`
Использовано строк: {data_rows}
Ограничение `--max-rows`: {max_rows}

## Линейная регрессия

- Train R2: {linear_metrics.train_r2:.6f}
- Test R2: {linear_metrics.test_r2:.6f}
- Test MSE: {linear_metrics.test_mse:.6f}
- Test MAE: {linear_metrics.test_mae:.6f}

## Полиномиальная регрессия

Лучшая степень по Test R2: {int(best_polynomial["degree"])}

- Train R2: {best_polynomial["train_r2"]:.6f}
- Test R2: {best_polynomial["test_r2"]:.6f}
- Test MSE: {best_polynomial["test_mse"]:.6f}
- Test MAE: {best_polynomial["test_mae"]:.6f}

## Ridge-регуляризация

Степень полинома для Ridge: {int(best_ridge["degree"])}
Лучший alpha по Test R2: {best_ridge["alpha"]:.6g}

- Train R2: {best_ridge["train_r2"]:.6f}
- Test R2: {best_ridge["test_r2"]:.6f}
- Test MSE: {best_ridge["test_mse"]:.6f}
- Test MAE: {best_ridge["test_mae"]:.6f}

## Выходные файлы

- `metrics.csv`
- `polynomial_results.csv`
- `ridge_results.csv`
- `polynomial_degree_r2.png`
- `ridge_alpha_r2.png`
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab03 variant 12: linear regression experiments for UCI power consumption data."
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20000,
        help="Number of rows sampled uniformly from the cleaned dataset. Use 0 for all rows.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction for manual split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for manual train/test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset(max_rows=args.max_rows)
    x = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = data[TARGET_COLUMN].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = manual_train_test_split(
        x,
        y,
        test_size=args.test_size,
        seed=args.seed,
    )

    print(f"Rows used: {len(data)}")
    print(f"Train rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")

    linear_metrics = build_linear_model(x_train, x_test, y_train, y_test)
    print(
        "LinearRegression: "
        f"train R2={linear_metrics.train_r2:.4f}, "
        f"test R2={linear_metrics.test_r2:.4f}, "
        f"test MSE={linear_metrics.test_mse:.4f}, "
        f"test MAE={linear_metrics.test_mae:.4f}"
    )

    polynomial_results = polynomial_experiments(
        x_train,
        x_test,
        y_train,
        y_test,
        degrees=[1, 2, 3, 4, 5],
    )
    best_degree = int(
        polynomial_results.sort_values("test_r2", ascending=False).iloc[0]["degree"]
    )

    ridge_results = ridge_experiments(
        x_train,
        x_test,
        y_train,
        y_test,
        alphas=np.logspace(-4, 4, 17),
        degree=best_degree,
    )

    metrics = pd.DataFrame(
        [
            linear_metrics.__dict__,
            polynomial_results.sort_values("test_r2", ascending=False)
            .iloc[0]
            .drop(labels=["degree"])
            .to_dict(),
            ridge_results.sort_values("test_r2", ascending=False)
            .iloc[0]
            .drop(labels=["alpha", "degree"])
            .to_dict(),
        ]
    )

    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    polynomial_results.to_csv(OUTPUT_DIR / "polynomial_results.csv", index=False)
    ridge_results.to_csv(OUTPUT_DIR / "ridge_results.csv", index=False)
    save_polynomial_plot(polynomial_results)
    save_ridge_plot(ridge_results)
    save_report(
        linear_metrics=linear_metrics,
        polynomial_results=polynomial_results,
        ridge_results=ridge_results,
        data_rows=len(data),
        max_rows=args.max_rows,
    )

    print(f"Saved results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
