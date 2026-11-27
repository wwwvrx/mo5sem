# Лабораторная работа №3, вариант 12

Вариант: **Individual household electric power consumption** из UCI Machine Learning Repository.

Цель регрессии в решении: `Global_active_power`.

Используемые признаки:

- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`
- `hour`
- `day_of_week`
- `month`

Скрипт выполняет все пункты из [Lab03.md](Lab03.md):

1. загружает датасет UCI, если файл еще не лежит в `data/`;
2. очищает пропуски и добавляет календарные признаки;
3. вручную разделяет выборку на обучающую и тестовую без `train_test_split`;
4. обучает обычную `LinearRegression`;
5. строит полиномиальные модели разных степеней;
6. подбирает коэффициент регуляризации для `Ridge`;
7. сохраняет графики и CSV с результатами в `outputs/`.

## Запуск

```bash
cd Lab03_variant12
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python lab03_variant12.py
```

Источник датасета: <https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption>

По умолчанию берется равномерная подвыборка из `20000` строк, чтобы полиномиальные модели обучались быстро. Полный датасет содержит больше 2 млн записей, поэтому для экспериментов можно менять размер подвыборки:

```bash
python lab03_variant12.py --max-rows 120000
```

Если датасет уже скачан вручную, положите `household_power_consumption.txt` в папку `data/`.
