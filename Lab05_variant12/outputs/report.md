# Результаты лабораторной работы №5, вариант 12

Датасет: NYSK
Источник: https://archive.ics.uci.edu/dataset/260/nysk
Использовано документов: 3000
Размер словаря TF-IDF: 5000
Компонент TruncatedSVD: 80
Доля объясненной дисперсии SVD: 0.352241

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

- Алгоритм: KMeans
- Параметры: {'n_clusters': 12.0}
- Количество кластеров: 12
- Шумовых объектов: 0
- Доля крупнейшего кластера: 0.482333
- Доля шумовых объектов: 0.000000
- Silhouette: 0.059816
- Calinski-Harabasz: 35.344584
- Davies-Bouldin: 3.061254

## Размеры кластеров лучшей модели

0     1447
1       16
2       16
3       45
4      508
5       40
6      170
7       32
8      436
9       42
10     231
11      17

## Выходные файлы

- `documents.csv`
- `clustering_results.csv`
- `best_cluster_assignments.csv`
- `cluster_examples.csv`
- `silhouette_by_clusters.png`
- `algorithm_best_silhouette.png`
- `dbscan_eps_clusters.png`
- `best_cluster_sizes.png`
