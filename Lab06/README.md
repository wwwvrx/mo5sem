# Лабораторная работа №6. Сверточные сети

Датасет: **Fashion-MNIST** из `keras.datasets`.

В `Lab06.md` нет списка вариантов, поэтому выбран стандартный учебный набор изображений для классификации. Fashion-MNIST содержит 10 классов одежды, изображения `28x28` в оттенках серого и хорошо подходит для демонстрации сверточной сети.

Скрипты выполняют задание:

1. `train_cnn.py` загружает датасет через Keras, создает CNN, обучает ее, проверяет точность и сохраняет модель в `.keras` файл.
2. `predict_image.py` загружает сохраненную модель и классифицирует изображение из файла или пример из тестовой выборки.

Используется Keras 3 с backend `torch`, чтобы код работал через современный Keras API без привязки к конкретной версии TensorFlow.

## Запуск

```bash
cd Lab06
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_cnn.py
```

Модель сохраняется в:

```text
outputs/fashion_mnist_cnn.keras
```

## Проверка классификатора

Проверить на тестовом примере Fashion-MNIST:

```bash
python predict_image.py --sample-index 0
```

Проверить на своем изображении:

```bash
python predict_image.py --image path/to/image.png
```

Для быстрого запуска по умолчанию используется подвыборка из `12000` обучающих изображений и `3000` тестовых. Можно увеличить:

```bash
python train_cnn.py --train-limit 60000 --test-limit 10000 --epochs 5
```
