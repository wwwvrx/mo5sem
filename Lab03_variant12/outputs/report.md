# Результаты лабораторной работы №3, вариант 12

Датасет: Individual household electric power consumption
Целевая переменная: `Global_active_power`
Использовано строк: 20000
Ограничение `--max-rows`: 20000

## Линейная регрессия

- Train R2: 0.998512
- Test R2: 0.998234
- Test MSE: 0.001960
- Test MAE: 0.026887

## Полиномиальная регрессия

Лучшая степень по Test R2: 3

- Train R2: 0.998992
- Test R2: 0.998778
- Test MSE: 0.001356
- Test MAE: 0.022016

## Ridge-регуляризация

Степень полинома для Ridge: 3
Лучший alpha по Test R2: 1

- Train R2: 0.998983
- Test R2: 0.998781
- Test MSE: 0.001353
- Test MAE: 0.021981

## Выходные файлы

- `metrics.csv`
- `polynomial_results.csv`
- `ridge_results.csv`
- `polynomial_degree_r2.png`
- `ridge_alpha_r2.png`
