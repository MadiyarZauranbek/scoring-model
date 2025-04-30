# credit_scoring/models/logistic_model.py
import numpy as np
from .base_model import BaseModel
from ..optim.optimizers import GradientDescentOptimizer

# Устанавливаем обработку ошибок NumPy для предотвращения некоторых предупреждений
# np.seterr(over='raise', divide='raise') # Можно раскомментировать для отладки, но может прерывать работу

class LogisticModel(BaseModel):
    """
    Модель логистической регрессии с L2-регуляризацией,
    обучаемая с помощью градиентного спуска.
    """
    def __init__(self,
                 learning_rate: float = 0.01,
                 iterations: int = 1000,
                 lambda_reg: float = 0.1,
                 tolerance: float = 1e-5, # Для возможной ранней остановки
                 verbose: bool = True):   # Выводить лог обучения?
        """
        Инициализация модели.

        :param learning_rate: Скорость обучения для градиентного спуска.
        :param iterations: Максимальное количество итераций градиентного спуска.
        :param lambda_reg: Коэффициент L2-регуляризации (0 - нет регуляризации).
        :param tolerance: Порог изменения функции потерь для ранней остановки.
        :param verbose: Если True, выводит информацию о ходе обучения.
        """
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("learning_rate должен быть положительным числом")
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations должно быть положительным целым числом")
        if not isinstance(lambda_reg, (int, float)) or lambda_reg < 0:
            raise ValueError("lambda_reg должен быть неотрицательным числом")
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            raise ValueError("tolerance должен быть положительным числом")

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.tolerance = tolerance
        self.verbose = verbose

        # Оптимизатор теперь просто хранит learning_rate
        # self.optimizer = GradientDescentOptimizer(learning_rate=self.learning_rate) # Можно и так
        self.weights = None # Веса модели (включая bias/intercept), инициализируются в fit
        self.history = {'loss': []} # Для хранения истории функции потерь

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Добавляет столбец единиц к матрице X для учета свободного члена (bias/intercept)."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Сигмоидная функция."""
        # Добавим клиппинг для предотвращения переполнения в np.exp
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def _compute_cost(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Вычисляет функцию потерь (бинарная кросс-энтропия + L2 регуляризация)."""
        m = len(y)
        if m == 0: return 0.0

        z = X @ weights
        h = self._sigmoid(z)

        # Добавим epsilon для численной стабильности логарифма
        epsilon = 1e-8
        cost = (-1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

        # L2 регуляризация (не регуляризуем свободный член weights[0])
        l2_reg_cost = (self.lambda_reg / (2 * m)) * np.sum(weights[1:]**2)

        total_cost = cost + l2_reg_cost
        return total_cost

    def _compute_gradient(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Вычисляет градиент функции потерь по весам."""
        m = len(y)
        if m == 0: return np.zeros_like(weights)

        z = X @ weights
        h = self._sigmoid(z)
        error = h - y

        # Градиент без регуляризации
        gradient = (1 / m) * (X.T @ error)

        # Добавляем градиент от L2 регуляризации (кроме свободного члена weights[0])
        l2_reg_gradient = (self.lambda_reg / m) * weights
        l2_reg_gradient[0] = 0 # Не регуляризуем bias

        total_gradient = gradient + l2_reg_gradient
        return total_gradient

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Обучает модель логистической регрессии.

        :param X_train: Массив признаков обучающей выборки.
        :param y_train: Вектор целевой переменной обучающей выборки.
        """
        if X_train.ndim == 1:
             X_train = X_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1) # Убедимся, что y - вектор-столбец

        if X_train.shape[0] != y_train.shape[0]:
             raise ValueError(f"Несовпадение количества образцов в X_train ({X_train.shape[0]}) и y_train ({y_train.shape[0]})")

        print(f"Starting training Logistic Regression model...")
        print(f"Params: LR={self.learning_rate}, Iterations={self.iterations}, Lambda={self.lambda_reg}")

        X_train_int = self._add_intercept(X_train)
        n_samples, n_features = X_train_int.shape

        # Инициализация весов нулями
        self.weights = np.zeros((n_features, 1)) # Веса теперь вектор-столбец
        self.history['loss'] = [] # Очищаем историю потерь

        for i in range(self.iterations):
            # Рассчитываем градиент
            gradients = self._compute_gradient(X_train_int, y_train, self.weights)

            # Обновляем веса (шаг градиентного спуска)
            # Используем напрямую learning_rate, т.к. оптимизатор его просто хранит
            self.weights = self.weights - self.learning_rate * gradients

            # Рассчитываем и сохраняем значение функции потерь
            cost = self._compute_cost(X_train_int, y_train, self.weights)
            self.history['loss'].append(cost)

            # Логирование прогресса
            if self.verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{self.iterations}, Loss: {cost:.6f}")

            # Проверка на раннюю остановку (если изменение loss меньше tolerance)
            if i > 0 and abs(self.history['loss'][-1] - self.history['loss'][-2]) < self.tolerance:
                print(f"Early stopping at iteration {i + 1} due to tolerance.")
                break

        print("Training finished.")
        print(f"Final loss: {self.history['loss'][-1]:.6f}")


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов.

        :param X: Массив признаков для предсказания.
        :return: Массив [n_samples, 2] с вероятностями [класс 0, класс 1].
        """
        if self.weights is None:
            raise RuntimeError("Модель еще не обучена. Вызовите fit() перед предсказанием.")
        if X.ndim == 1:
             X = X.reshape(-1, 1)

        X_int = self._add_intercept(X)
        z = X_int @ self.weights
        proba_pos = self._sigmoid(z) # Вероятность класса 1
        proba_neg = 1 - proba_pos    # Вероятность класса 0
        # Возвращаем массив [n_samples, 2]
        return np.concatenate((proba_neg, proba_pos), axis=1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Предсказывает метки классов (0 или 1).

        :param X: Массив признаков для предсказания.
        :param threshold: Порог для отнесения к классу 1.
        :return: Вектор предсказанных классов [n_samples,].
        """
        if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
            raise ValueError("threshold должен быть числом между 0 и 1")

        # Получаем вероятности для класса 1 (второй столбец)
        probabilities = self.predict_proba(X)[:, 1]
        # Применяем порог и преобразуем в int (0 или 1)
        return (probabilities >= threshold).astype(int)

    # --- Дополнительно: Методы для сохранения/загрузки модели ---
    def save_model(self, file_path: str):
        """Сохраняет веса модели в файл."""
        if self.weights is None:
            raise RuntimeError("Модель не обучена, нечего сохранять.")
        # Можно сохранять и другие параметры, если нужно
        artefacts = {'weights': self.weights}
        with open(file_path, 'wb') as f:
            np.save(f, artefacts) # Используем np.save для словаря с numpy array
        print(f"Model weights saved to {file_path}")

    @classmethod
    def load_model(cls, file_path: str, **kwargs):
        """
        Загружает веса модели из файла и создает экземпляр класса.
        kwargs можно использовать для передачи гиперпараметров (lr, iter и т.д.)
        если они не сохранялись вместе с весами.
        """
        with open(file_path, 'rb') as f:
            # allow_pickle=True нужно т.к. сохраняли словарь
            artefacts = np.load(f, allow_pickle=True).item()

        # Создаем экземпляр класса с переданными гиперпараметрами
        model = cls(**kwargs)
        # Загружаем веса
        model.weights = artefacts['weights']
        print(f"Model weights loaded from {file_path}")
        return model