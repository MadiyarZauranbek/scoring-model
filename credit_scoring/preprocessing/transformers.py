# credit_scoring/preprocessing/transformers.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformer:
    """
    Класс для комплексной предобработки данных:
    - Заполнение пропусков (числовые - медианой, категориальные - модой)
    - One-Hot кодирование категориальных признаков
    - Масштабирование числовых признаков (StandardScaler)
    """
    def __init__(self, numerical_cols: list, categorical_cols: list):
        """
        :param numerical_cols: Список названий числовых колонок.
        :param categorical_cols: Список названий категориальных колонок.
        """
        if not isinstance(numerical_cols, list) or not isinstance(categorical_cols, list):
             raise TypeError("numerical_cols и categorical_cols должны быть списками")
             
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.preprocessor = self._build_preprocessor()
        self._is_fitted = False # Флаг, что трансформер обучен
        self.feature_names_out_ = None # Инициализируем атрибут для имен признаков

    def _build_preprocessor(self) -> ColumnTransformer:
        """Создает объект ColumnTransformer для всех шагов."""

        # Шаги для числовых признаков: Заполнение + Масштабирование
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Шаги для категориальных признаков: Заполнение + OHE
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # sparse_output=False чтобы получить numpy array, а не разреженную матрицу
            # handle_unknown='ignore' чтобы при transform не падать на новые категории
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Собираем все вместе в ColumnTransformer
        # remainder='passthrough' означает, что колонки, не указанные
        # в numerical_cols или categorical_cols, останутся в данных без изменений.
        # Если все колонки должны быть обработаны, можно использовать remainder='drop'
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='passthrough'
        )
        return preprocessor

    def fit(self, X: pd.DataFrame):
        """
        Обучает препроцессор на данных X.
        Запоминает параметры (средние, категории и т.д.).
        :param X: DataFrame с данными для обучения препроцессора.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("На вход методу fit должен подаваться pandas DataFrame")

        print(f"Fitting DataTransformer on {X.shape[0]} samples, {X.shape[1]} features.")
        print(f"Numerical columns: {self.numerical_cols}")
        print(f"Categorical columns: {self.categorical_cols}")

        # Проверяем наличие всех необходимых колонок в DataFrame
        all_needed_cols = self.numerical_cols + self.categorical_cols
        missing_cols = [col for col in all_needed_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Следующие колонки не найдены в DataFrame: {missing_cols}")

        self.preprocessor.fit(X)
        self._is_fitted = True
        print("DataTransformer fitted successfully.")

        # Пытаемся получить имена признаков после обработки (особенно важно после OHE)
        try:
            # Этот метод появился в sklearn 1.0 и является предпочтительным
            self.feature_names_out_ = self.preprocessor.get_feature_names_out()
            # print(f"Output feature names generated: {list(self.feature_names_out_)}")
        except Exception as e:
            print(f"Warning: Could not automatically get feature names via get_feature_names_out: {e}")
            # Если метод не сработал, оставляем None, get_feature_names попробует резервный вариант
            self.feature_names_out_ = None

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Применяет обученный препроцессор к данным X.
        :param X: DataFrame с данными для трансформации.
        :return: NumPy массив с обработанными признаками.
        """
        if not self._is_fitted:
            raise RuntimeError("Препроцессор еще не обучен. Вызовите fit() перед transform().")
        if not isinstance(X, pd.DataFrame):
             raise ValueError("На вход методу transform должен подаваться pandas DataFrame")

        # Проверяем наличие всех необходимых колонок
        all_needed_cols = self.numerical_cols + self.categorical_cols
        missing_cols = [col for col in all_needed_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Следующие колонки не найдены в DataFrame для трансформации: {missing_cols}")

        print(f"Transforming data with {X.shape[0]} samples.")
        X_processed = self.preprocessor.transform(X)
        print(f"Transformation complete. Output shape: {X_processed.shape}")
        return X_processed

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Обучает препроцессор и сразу применяет его к данным X.
        :param X: DataFrame с данными для обучения и трансформации.
        :return: NumPy массив с обработанными признаками.
        """
        # Просто последовательно вызываем fit и transform
        return self.fit(X).transform(X)

    def get_feature_names(self) -> list:
        """
        Возвращает имена признаков после обработки.
        Предпочитает использовать метод get_feature_names_out(), если он сработал в fit.
        Иначе пытается собрать имена вручную (менее надежно).
        """
        if not self._is_fitted:
            raise RuntimeError("Препроцессор не обучен. Невозможно получить имена признаков.")

        # --- Блок 1: Основной (предпочтительный) способ ---
        if self.feature_names_out_ is not None:
            # Если имена были успешно получены в fit(), возвращаем их
            return list(self.feature_names_out_)

        # --- Блок 2: Резервный (менее надежный) способ ---
        else:
            print("Warning: Falling back to manual feature name generation (less reliable).")
            try:
                # Получаем имена от OHE-кодировщика
                ohe_feature_names = self.preprocessor.named_transformers_['cat'] \
                                        .named_steps['onehot'] \
                                        .get_feature_names_out(self.categorical_cols)
                # Числовые колонки StandardScaler не переименовывает
                num_feature_names = self.numerical_cols

                # Собираем имена числовых и категориальных признаков
                # ВНИМАНИЕ: Эта резервная логика НЕ обрабатывает колонки из 'remainder'!
                # Если использовался remainder='passthrough', имена этих колонок здесь НЕ будут учтены.
                generated_names = list(num_feature_names) + list(ohe_feature_names)
                print(f"Manually generated names (numerical + categorical only): {generated_names}")
                return generated_names
            except Exception as e:
                 # Ловим конкретную ошибку, а не все подряд
                 print(f"Error during manual feature name generation: {e}")
                 print("Warning: Could not reliably determine feature names.")
                 return []