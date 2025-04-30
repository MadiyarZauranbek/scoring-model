# credit_scoring/optim/optimizers.py
import numpy as np

class GradientDescentOptimizer:
    """Оптимизатор методом градиентного спуска."""

    def __init__(self, learning_rate: float = 0.01):
        """
        :param learning_rate: Скорость обучения (шаг градиентного спуска).
        # lambda_reg (сила регуляризации) и iterations (кол-во итераций)
        # лучше хранить в самой модели, т.к. они относятся к процессу обучения модели,
        # а оптимизатор лишь выполняет один шаг обновления весов.
        """
        if learning_rate <= 0:
            raise ValueError("Learning rate должен быть положительным числом.")
        self.learning_rate = learning_rate

    def step(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Выполняет один шаг градиентного спуска.

        :param weights: Текущие веса модели (включая bias/intercept).
        :param gradients: Градиенты функции потерь по весам (рассчитанные в модели).
                          Важно: модель сама должна добавить в градиент компоненту от регуляризации.
        :return: Обновленные веса модели.
        """
        if weights.shape != gradients.shape:
             raise ValueError(f"Размерности весов {weights.shape} и градиентов {gradients.shape} не совпадают")

        updated_weights = weights - self.learning_rate * gradients
        return updated_weights

# Можно добавить и другие оптимизаторы, например, SGD, Adam, если потребуется.
# class SGD(GradientDescentOptimizer): # Пример наследования
#     def __init__(self, learning_rate: float = 0.01, batch_size: int = 32):
#         super().__init__(learning_rate)
#         self.batch_size = batch_size
#         # Логика SGD будет отличаться в методе step или в цикле обучения модели