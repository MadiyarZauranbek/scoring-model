# Библиотека Скоринговой Модели

Проект по разработке Python-библиотеки для предсказания кредитного риска клиента банка на основе исторических данных. Библиотека реализована с использованием ООП и включает собственную реализацию логистической регрессии с L2-регуляризацией и градиентным спуском.

## Структура проекта

- `credit_scoring/`: Основной пакет библиотеки.
    - `models/`: Реализация моделей (`BaseModel`, `LogisticModel`).
    - `optim/`: Оптимизатор градиентного спуска.
    - `preprocessing/`: Класс для предобработки данных (`DataTransformer`).
    - `metrics/`: Функции для расчета метрик качества.
- `data/`: Папка для размещения данных (например, `train-scoring.csv`).
- `train.py`: Скрипт для обучения модели, кросс-валидации и оценки.
- `predict.py`: Скрипт для предсказания на новых данных с использованием обученной модели.
- `config.json`: Файл конфигурации (гиперпараметры, пути, колонки).
- `requirements.txt`: Список зависимостей Python.
- `*.pkl`: Сохраненные артефакты (обученная модель, трансформер).
- `*.png`: Сохраненные графики (кривая обучения, ROC, матрица ошибок).
- `training.log`: Лог-файл процесса обучения.

## Установка и Настройка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone [https://github.com/MadiyarZauranbek/scoring-model.git](https://github.com/MadiyarZauranbek/scoring-model.git)
    cd scoring-model
    ```
2.  **Создайте и активируйте виртуальное окружение:**
    ```bash
    python -m venv venv
    # Windows PowerShell:
    .\venv\Scripts\Activate.ps1
    # Windows CMD:
    # .\venv\Scripts\activate
    # macOS / Linux:
    # source venv/bin/activate
    ```
3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Данные:** Поместите файл с обучающими данными (например, `train-scoring.csv`) в папку `data/`.
5.  **Настройте `config.json`:** Откройте `config.json` и убедитесь, что:
    * `train_params.data_path` указывает на ваш файл данных (например, `"data/train-scoring.csv"`).
    * Списки `preprocessing_params.numerical_cols` и `categorical_cols` содержат правильные имена колонок из вашего датасета.
    * `preprocessing_params.target_col` содержит имя целевой колонки.

## Использование

### Обучение модели

Запустите скрипт `train.py` из корневой папки проекта (с активированным venv):
```bash
python train.py