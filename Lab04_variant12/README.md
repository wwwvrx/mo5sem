# Лабораторная работа №4, вариант 12

Вариант: **sEMG for Basic Hand movements** из UCI Machine Learning Repository.

Источник датасета: <https://archive.ics.uci.edu/dataset/313/semg+for+basic+hand+movements>

Скрипт выполняет пункты из [Lab04.md](Lab04.md):

1. скачивает UCI-архив, если он еще не лежит в `data/`;
2. читает `.mat` файлы с sEMG-сигналами;
3. извлекает числовые признаки из двух каналов сигнала;
4. вручную делит данные на `train / validation / test` без `train_test_split`;
5. выполняет scaling через `StandardScaler`;
6. обучает `Perceptron` и `MLPClassifier`;
7. проводит эксперименты с коэффициентом обучения, регуляризацией и solver;
8. сохраняет CSV, графики и отчет в `outputs/`.

## Запуск

```bash
cd Lab04_variant12
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python lab04_variant12.py
```

По умолчанию используется весь набор файлов из UCI-архива. Можно ограничиться первой или второй базой:

```bash
python lab04_variant12.py --database database1
python lab04_variant12.py --database database2
```
