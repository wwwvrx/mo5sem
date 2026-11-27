from __future__ import annotations

import argparse
import os
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

MPLCONFIGDIR = Path(__file__).with_name(".matplotlib")
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


DATASET_URLS = (
    "https://archive.ics.uci.edu/static/public/260/nysk.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00260/nysk.zip",
)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
RAW_ZIP = DATA_DIR / "nysk.zip"
XML_FILE = DATA_DIR / "nysk.xml"


@dataclass(frozen=True)
class FeatureData:
    documents: pd.DataFrame
    matrix: np.ndarray
    vocabulary_size: int
    explained_variance: float


def clean_tag(tag: str) -> str:
    return tag.split("}", maxsplit=1)[-1].lower()


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if XML_FILE.exists():
        return

    last_error: Exception | None = None
    for url in DATASET_URLS:
        try:
            print(f"Downloading dataset: {url}")
            with urllib.request.urlopen(url, timeout=180) as response:
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
            f"place nysk.xml into {DATA_DIR.resolve()}."
        ) from last_error

    with zipfile.ZipFile(RAW_ZIP) as archive:
        archive.extractall(DATA_DIR)

    found_xml = list(DATA_DIR.rglob("nysk.xml"))
    if not found_xml:
        raise RuntimeError(f"Archive was downloaded, but nysk.xml was not found in {DATA_DIR.resolve()}.")
    if found_xml[0] != XML_FILE:
        shutil.move(str(found_xml[0]), XML_FILE)


def extract_leaf_texts(element: ET.Element) -> dict[str, str]:
    values: dict[str, list[str]] = {}
    for child in element.iter():
        if child is element:
            continue
        text = " ".join("".join(child.itertext()).split())
        if not text:
            continue
        tag = clean_tag(child.tag)
        values.setdefault(tag, []).append(text)
    return {tag: " ".join(texts) for tag, texts in values.items()}


def parse_documents(max_docs: int) -> pd.DataFrame:
    download_dataset()

    records: list[dict[str, str | int]] = []
    context = ET.iterparse(XML_FILE, events=("end",))

    for _, element in context:
        if clean_tag(element.tag) != "document":
            continue

        fields = extract_leaf_texts(element)
        text = fields.get("text", "")
        title = fields.get("title", "")
        if not text and title:
            text = title

        text = normalize_whitespace(text)
        title = normalize_whitespace(title)
        if text:
            records.append(
                {
                    "document_id": len(records),
                    "title": title,
                    "date": fields.get("date", ""),
                    "source": fields.get("source", ""),
                    "text": text,
                    "text_length": len(text),
                    "word_count": len(text.split()),
                }
            )

        element.clear()

    documents = pd.DataFrame(records)
    if documents.empty:
        raise RuntimeError("No <document> records with text were found in nysk.xml.")

    if max_docs > 0 and len(documents) > max_docs:
        indices = np.linspace(0, len(documents) - 1, max_docs, dtype=int)
        documents = documents.iloc[indices].reset_index(drop=True)

    return documents


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_feature_matrix(
    documents: pd.DataFrame,
    max_features: int,
    svd_components: int,
    seed: int,
) -> FeatureData:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_df=0.85,
        min_df=3,
        max_features=max_features,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(documents["text"])

    component_count = min(svd_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
    if component_count < 2:
        raise RuntimeError("Not enough TF-IDF features to run TruncatedSVD.")

    svd = TruncatedSVD(n_components=component_count, random_state=seed)
    reduced = svd.fit_transform(tfidf)
    scaled = StandardScaler().fit_transform(reduced)

    return FeatureData(
        documents=documents,
        matrix=scaled,
        vocabulary_size=len(vectorizer.vocabulary_),
        explained_variance=float(np.sum(svd.explained_variance_ratio_)),
    )


def evaluate_labels(
    features: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    params: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    labels = np.asarray(labels)
    cluster_count = len(set(labels) - {-1})
    noise_count = int(np.sum(labels == -1))
    non_noise = labels[labels != -1]
    if len(non_noise):
        largest_cluster_share = max(Counter(non_noise).values()) / len(labels)
    else:
        largest_cluster_share = 1.0
    noise_share = noise_count / len(labels)

    row: dict[str, float | int | str] = {
        "algorithm": algorithm,
        **params,
        "clusters": cluster_count,
        "noise_points": noise_count,
        "largest_cluster_share": largest_cluster_share,
        "noise_share": noise_share,
    }

    valid_for_metrics = 1 < len(np.unique(labels)) < len(labels)
    if valid_for_metrics:
        row["silhouette"] = float(silhouette_score(features, labels))
        row["calinski_harabasz"] = float(calinski_harabasz_score(features, labels))
        row["davies_bouldin"] = float(davies_bouldin_score(features, labels))
    else:
        row["silhouette"] = np.nan
        row["calinski_harabasz"] = np.nan
        row["davies_bouldin"] = np.nan

    return row


def run_kmeans(features: np.ndarray, cluster_values: list[int], seed: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for n_clusters in cluster_values:
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
        labels = model.fit_predict(features)
        row = evaluate_labels(features, labels, "KMeans", {"n_clusters": n_clusters})
        rows.append(row)
        print(
            f"KMeans k={n_clusters}: silhouette={row['silhouette']:.4f}, "
            f"clusters={row['clusters']}"
        )
    return rows


def run_agglomerative(features: np.ndarray, cluster_values: list[int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for n_clusters in cluster_values:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(features)
        row = evaluate_labels(
            features,
            labels,
            "AgglomerativeClustering",
            {"n_clusters": n_clusters, "linkage": "ward"},
        )
        rows.append(row)
        print(
            f"Agglomerative k={n_clusters}: silhouette={row['silhouette']:.4f}, "
            f"clusters={row['clusters']}"
        )
    return rows


def run_birch(features: np.ndarray, cluster_values: list[int]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for n_clusters in cluster_values:
        model = Birch(n_clusters=n_clusters, threshold=0.7)
        labels = model.fit_predict(features)
        row = evaluate_labels(
            features,
            labels,
            "Birch",
            {"n_clusters": n_clusters, "threshold": 0.7},
        )
        rows.append(row)
        print(
            f"Birch k={n_clusters}: silhouette={row['silhouette']:.4f}, "
            f"clusters={row['clusters']}"
        )
    return rows


def run_dbscan(features: np.ndarray, eps_values: list[float]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(features)
        row = evaluate_labels(
            features,
            labels,
            "DBSCAN",
            {"eps": eps, "min_samples": 5},
        )
        rows.append(row)
        silhouette_value = row["silhouette"]
        printable = "nan" if pd.isna(silhouette_value) else f"{silhouette_value:.4f}"
        print(
            f"DBSCAN eps={eps:g}: silhouette={printable}, "
            f"clusters={row['clusters']}, noise={row['noise_points']}"
        )
    return rows


def run_experiments(features: np.ndarray, seed: int) -> pd.DataFrame:
    cluster_values = [4, 6, 8, 10, 12, 16]
    rows: list[dict[str, float | int | str]] = []
    rows.extend(run_kmeans(features, cluster_values, seed))
    rows.extend(run_agglomerative(features, cluster_values))
    rows.extend(run_birch(features, cluster_values))
    rows.extend(run_dbscan(features, [1.5, 2.0, 2.5, 3.0, 3.5]))
    return pd.DataFrame(rows)


def choose_best_result(results: pd.DataFrame) -> pd.Series:
    scored = results.dropna(subset=["silhouette"]).copy()
    if scored.empty:
        raise RuntimeError("No valid clustering result was produced.")

    practical = scored[
        (scored["largest_cluster_share"] <= 0.75)
        & (scored["noise_share"] <= 0.50)
        & (scored["clusters"] >= 3)
    ].copy()
    if practical.empty:
        practical = scored.copy()

    return practical.sort_values(
        ["silhouette", "calinski_harabasz", "davies_bouldin"],
        ascending=[False, False, True],
    ).iloc[0]


def fit_best_model(features: np.ndarray, best: pd.Series, seed: int) -> np.ndarray:
    algorithm = best["algorithm"]
    if algorithm == "KMeans":
        model = KMeans(n_clusters=int(best["n_clusters"]), n_init=20, random_state=seed)
    elif algorithm == "AgglomerativeClustering":
        model = AgglomerativeClustering(n_clusters=int(best["n_clusters"]), linkage="ward")
    elif algorithm == "Birch":
        model = Birch(n_clusters=int(best["n_clusters"]), threshold=float(best["threshold"]))
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=float(best["eps"]), min_samples=int(best["min_samples"]))
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return model.fit_predict(features)


def save_metric_plot(results: pd.DataFrame) -> None:
    clustered = results[results["algorithm"] != "DBSCAN"].dropna(subset=["silhouette"])
    plt.figure(figsize=(10, 5))
    for algorithm, group in clustered.groupby("algorithm"):
        group = group.sort_values("n_clusters")
        plt.plot(group["n_clusters"], group["silhouette"], marker="o", label=algorithm)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("Clustering quality by cluster count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "silhouette_by_clusters.png", dpi=160)
    plt.close()


def save_dbscan_plot(results: pd.DataFrame) -> None:
    dbscan = results[results["algorithm"] == "DBSCAN"].copy()
    if dbscan.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(dbscan["eps"], dbscan["clusters"], marker="o", label="clusters")
    plt.plot(dbscan["eps"], dbscan["noise_points"], marker="o", label="noise points")
    plt.xlabel("eps")
    plt.ylabel("Count")
    plt.title("DBSCAN parameter experiment")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dbscan_eps_clusters.png", dpi=160)
    plt.close()


def save_algorithm_bar(results: pd.DataFrame) -> None:
    grouped = (
        results.dropna(subset=["silhouette"])
        .groupby("algorithm", as_index=False)["silhouette"]
        .max()
        .sort_values("silhouette", ascending=False)
    )
    plt.figure(figsize=(9, 5))
    plt.bar(grouped["algorithm"], grouped["silhouette"])
    plt.xlabel("Algorithm")
    plt.ylabel("Best silhouette score")
    plt.title("Best clustering result by algorithm")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "algorithm_best_silhouette.png", dpi=160)
    plt.close()


def save_cluster_size_plot(labels: np.ndarray) -> None:
    counts = Counter(labels)
    ordered = sorted(counts.items(), key=lambda item: item[0])
    labels_text = ["noise" if cluster == -1 else str(cluster) for cluster, _ in ordered]
    values = [count for _, count in ordered]

    plt.figure(figsize=(10, 5))
    plt.bar(labels_text, values)
    plt.xlabel("Cluster")
    plt.ylabel("Documents")
    plt.title("Best model cluster sizes")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_cluster_sizes.png", dpi=160)
    plt.close()


def save_cluster_examples(documents: pd.DataFrame, labels: np.ndarray) -> None:
    examples = documents.copy()
    examples["cluster"] = labels
    examples["snippet"] = examples["text"].str.slice(0, 240)
    examples = (
        examples.sort_values(["cluster", "word_count"], ascending=[True, False])
        .groupby("cluster")
        .head(5)
    )
    examples[["cluster", "document_id", "date", "source", "title", "word_count", "snippet"]].to_csv(
        OUTPUT_DIR / "cluster_examples.csv",
        index=False,
    )


def save_report(
    feature_data: FeatureData,
    results: pd.DataFrame,
    best: pd.Series,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> None:
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    best_params = {
        key: value
        for key, value in best.to_dict().items()
        if key
        not in {
            "algorithm",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        "clusters",
        "noise_points",
        "largest_cluster_share",
        "noise_share",
        }
        and not pd.isna(value)
    }

    report = f"""# Результаты лабораторной работы №5, вариант 12

Датасет: NYSK
Источник: https://archive.ics.uci.edu/dataset/260/nysk
Использовано документов: {len(feature_data.documents)}
Размер словаря TF-IDF: {feature_data.vocabulary_size}
Компонент TruncatedSVD: {args.svd_components}
Доля объясненной дисперсии SVD: {feature_data.explained_variance:.6f}

## Предобработка

1. XML-документы извлекаются из тегов `<document>`.
2. Поле `text` преобразуется в TF-IDF с английскими stop words.
3. Размерность снижается через `TruncatedSVD`.
4. Признаки масштабируются через `StandardScaler`.

## Алгоритмы

Проверены алгоритмы:

- KMeans
- AgglomerativeClustering
- Birch
- DBSCAN

## Лучшая практическая модель по silhouette

Перед выбором итоговой модели отбрасываются вырожденные результаты, где больше 75% документов попали в один кластер или больше 50% документов помечены как шум.

- Алгоритм: {best["algorithm"]}
- Параметры: {best_params}
- Количество кластеров: {int(best["clusters"])}
- Шумовых объектов: {int(best["noise_points"])}
- Доля крупнейшего кластера: {best["largest_cluster_share"]:.6f}
- Доля шумовых объектов: {best["noise_share"]:.6f}
- Silhouette: {best["silhouette"]:.6f}
- Calinski-Harabasz: {best["calinski_harabasz"]:.6f}
- Davies-Bouldin: {best["davies_bouldin"]:.6f}

## Размеры кластеров лучшей модели

{cluster_counts.to_string()}

## Выходные файлы

- `documents.csv`
- `clustering_results.csv`
- `best_cluster_assignments.csv`
- `cluster_examples.csv`
- `silhouette_by_clusters.png`
- `algorithm_best_silhouette.png`
- `dbscan_eps_clusters.png`
- `best_cluster_sizes.png`
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lab05 variant 12: clustering experiments for the NYSK text dataset."
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=3000,
        help="Uniformly sampled number of XML documents. Use 0 for all documents.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--svd-components",
        type=int,
        default=80,
        help="Number of TruncatedSVD components before scaling and clustering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stochastic algorithms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = parse_documents(args.max_docs)
    print(f"Documents: {len(documents)}")

    feature_data = build_feature_matrix(
        documents,
        max_features=args.max_features,
        svd_components=args.svd_components,
        seed=args.seed,
    )
    print(f"TF-IDF vocabulary size: {feature_data.vocabulary_size}")
    print(f"SVD explained variance: {feature_data.explained_variance:.4f}")

    results = run_experiments(feature_data.matrix, args.seed)
    best = choose_best_result(results)
    labels = fit_best_model(feature_data.matrix, best, args.seed)

    assignments = feature_data.documents[["document_id", "date", "source", "title", "word_count"]].copy()
    assignments["cluster"] = labels

    feature_data.documents.to_csv(OUTPUT_DIR / "documents.csv", index=False)
    results.to_csv(OUTPUT_DIR / "clustering_results.csv", index=False)
    assignments.to_csv(OUTPUT_DIR / "best_cluster_assignments.csv", index=False)
    save_cluster_examples(feature_data.documents, labels)

    save_metric_plot(results)
    save_dbscan_plot(results)
    save_algorithm_bar(results)
    save_cluster_size_plot(labels)
    save_report(feature_data, results, best, labels, args)

    print("Best clustering result:")
    print(best.to_string())
    print(f"Saved results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
