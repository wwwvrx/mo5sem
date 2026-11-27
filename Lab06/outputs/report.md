# Результаты лабораторной работы №6

Датасет: Fashion-MNIST
Библиотека: Keras 3
Backend: torch
Размер обучающей подвыборки: 12000
Размер тестовой подвыборки: 3000
Количество эпох: 3
Batch size: 128

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

- Test loss: 0.502557
- Test accuracy: 0.820000

## Classification report

```text
              precision    recall  f1-score   support

 T-shirt/top       0.79      0.80      0.80       302
     Trouser       0.99      0.94      0.96       308
    Pullover       0.66      0.75      0.70       310
       Dress       0.80      0.87      0.83       298
        Coat       0.70      0.71      0.71       324
      Sandal       0.94      0.94      0.94       285
       Shirt       0.52      0.39      0.45       298
     Sneaker       0.90      0.93      0.92       293
         Bag       0.94      0.94      0.94       297
  Ankle boot       0.94      0.94      0.94       285

    accuracy                           0.82      3000
   macro avg       0.82      0.82      0.82      3000
weighted avg       0.82      0.82      0.82      3000

```

## Выходные файлы

- `fashion_mnist_cnn.keras`
- `training_history.json`
- `training_accuracy.png`
- `training_loss.png`
- `confusion_matrix.png`
- `sample_images/`
