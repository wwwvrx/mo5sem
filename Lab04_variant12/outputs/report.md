# Результаты лабораторной работы №4, вариант 12

Датасет: sEMG for Basic Hand movements
Источник: https://archive.ics.uci.edu/dataset/313/semg+for+basic+hand+movements
Использованная часть датасета: `all`
Количество объектов: 2700
Количество признаков после извлечения: 29
Классы: Cylindrical, Hook, Lateral, Palmar, Spherical, Tip

## Разделение выборки

- Train: 1620
- Validation: 540
- Test: 540

## Распределение классов

label
Cylindrical    450
Hook           450
Lateral        450
Palmar         450
Spherical      450
Tip            450

## Лучшая модель Perceptron

- eta0: 0.01
- alpha: 1e-06
- Train accuracy: 0.620370
- Validation accuracy: 0.579630
- Test accuracy: 0.596296

## Лучшая модель MLPClassifier

- solver: sgd
- learning_rate_init: 0.01
- alpha: 1e-05
- Train accuracy: 0.977160
- Validation accuracy: 0.922222
- Test accuracy: 0.905556

## Classification report для лучшего MLPClassifier на test

```text
              precision    recall  f1-score   support

 Cylindrical       0.96      0.99      0.97        90
        Hook       0.97      0.94      0.96        90
     Lateral       0.80      0.78      0.79        90
      Palmar       0.84      0.91      0.87        90
   Spherical       0.98      0.98      0.98        90
         Tip       0.89      0.83      0.86        90

    accuracy                           0.91       540
   macro avg       0.91      0.91      0.91       540
weighted avg       0.91      0.91      0.91       540

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
