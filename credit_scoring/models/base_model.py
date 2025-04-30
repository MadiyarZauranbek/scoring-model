# credit_scoring/models/base_model.py
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Абстрактный базовый класс для всех моделей."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучает модель на предоставленных данных.

        :param X: Массив признаков (обучающая выборка).
        :param y: Вектор целевой переменной (обучающая выборка).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания классов для входных данных.

        :param X: Массив признаков (данные для предсказания).
        :return: Вектор предсказанных классов.
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Делает предсказания вероятностей классов для входных данных.

        :param X: Массив признаков (данные для предсказания).
        :return: Массив вероятностей [n_samples, n_classes].
        """
        pass