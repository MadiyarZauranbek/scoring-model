# train.py
import json
import logging
import os
import pickle # Или можно import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve as sk_roc_curve
from sklearn.metrics import auc as sk_auc

# Импортируем наши модули
from credit_scoring.metrics import (accuracy_score, f1_score, precision_score,
                                    recall_score, roc_auc_score_func)
from credit_scoring.models.logistic_model import LogisticModel
from credit_scoring.preprocessing.transformers import DataTransformer

# --- Настройка Логирования ---
def setup_logging(log_config):
    """Настраивает базовую конфигурацию логирования."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(log_config.get('log_level', 'INFO').upper(), logging.INFO)
    log_file = log_config.get('log_file', 'training.log')

    # Убедимся, что директория для лога существует (если лог не в корне)
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file), # Вывод в файл
            logging.StreamHandler()        # Вывод в консоль
        ]
    )
    logging.info("Logging setup complete.")

# --- Загрузка Конфигурации ---
def load_config(config_path='config.json'):
    """Загружает конфигурационный файл."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        # Логируем важные параметры для проверки
        logging.debug(f"Config details: {json.dumps(config, indent=2)}")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {config_path}")
        raise

# --- Основная Функция Обучения ---
def run_training(config):
    """Выполняет полный цикл обучения, оценки и сохранения модели."""
    logging.info("Starting training process...")

    # --- 1. Загрузка и подготовка данных ---
    data_path = config['train_params']['data_path']
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}. Shape: {data.shape}")
    except FileNotFoundError:
        logging.error(f"Training data file not found at {data_path}")
        return # Прерываем выполнение, если нет данных

    target_col = config['preprocessing_params']['target_col']
    if target_col not in data.columns:
        logging.error(f"Target column '{target_col}' not found in the dataset.")
        return

    # Разделение на признаки (X) и цель (y)
    X = data.drop(columns=[target_col])
    y = data[target_col]
    logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logging.info(f"Target value distribution:\n{y.value_counts(normalize=True)}")

    # Разделение на обучающую и тестовую выборки
    test_split_ratio = config['train_params']['test_split_ratio']
    random_state = config['train_params']['random_state']
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_split_ratio,
            random_state=random_state,
            stratify=y # Важно для сохранения пропорций классов
        )
        logging.info(f"Data split into train/test sets. Train: {X_train.shape}, Test: {X_test.shape}")
    except Exception as e:
        logging.error(f"Error during train/test split: {e}")
        return


    # --- 2. Предобработка данных ---
    logging.info("Initializing data transformer...")
    num_cols = config['preprocessing_params']['numerical_cols']
    cat_cols = config['preprocessing_params']['categorical_cols']
    try:
        transformer = DataTransformer(numerical_cols=num_cols, categorical_cols=cat_cols)

        # Обучаем трансформер ТОЛЬКО на ОБУЧАЮЩЕЙ выборке
        logging.info("Fitting transformer on training data...")
        transformer.fit(X_train)

        # Применяем трансформер к обучающей и тестовой выборкам
        logging.info("Transforming training data...")
        X_train_processed = transformer.transform(X_train)
        logging.info("Transforming test data...")
        X_test_processed = transformer.transform(X_test)

        # Преобразуем y в numpy массив для единообразия
        y_train = y_train.to_numpy().reshape(-1, 1)
        y_test = y_test.to_numpy().reshape(-1, 1)

        logging.info(f"Data preprocessing complete. Processed train shape: {X_train_processed.shape}, Processed test shape: {X_test_processed.shape}")
        # Получаем имена признаков после обработки для логирования/отладки
        try:
            feature_names = transformer.get_feature_names()
            logging.debug(f"Features after transformation ({len(feature_names)}): {feature_names}")
        except Exception as e:
             logging.warning(f"Could not get feature names after transformation: {e}")

    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}", exc_info=True) # Добавим traceback в лог
        return


    # --- 3. Кросс-валидация ---
    logging.info("Starting cross-validation...")
    cv_folds = config['train_params']['cv_folds']
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}

    # Используем уже обработанные X_train_processed и y_train для CV
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_processed, y_train)):
        logging.info(f"--- CV Fold {fold + 1}/{cv_folds} ---")
        X_cv_train, X_cv_val = X_train_processed[train_idx], X_train_processed[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

        try:
            # Создаем НОВУЮ модель для каждого фолда
            cv_model = LogisticModel(
                learning_rate=config['model_params']['learning_rate'],
                iterations=config['model_params']['iterations'],
                lambda_reg=config['model_params']['lambda_reg'],
                verbose=False # Отключаем подробный вывод для CV
            )
            cv_model.fit(X_cv_train, y_cv_train)

            # Оценка на валидационном фолде
            y_pred_proba_val = cv_model.predict_proba(X_cv_val)
            y_pred_val = cv_model.predict(X_cv_val) # Используем порог 0.5 по умолчанию

            # Расчет метрик (y_cv_val нужно сделать 1D для метрик)
            y_cv_val_1d = y_cv_val.ravel()
            y_pred_val_1d = y_pred_val.ravel()
            proba_pos_val = y_pred_proba_val[:, 1] # Вероятности класса 1

            acc = accuracy_score(y_cv_val_1d, y_pred_val_1d)
            prec = precision_score(y_cv_val_1d, y_pred_val_1d)
            rec = recall_score(y_cv_val_1d, y_pred_val_1d)
            f1 = f1_score(y_cv_val_1d, y_pred_val_1d)
            roc_auc = roc_auc_score_func(y_cv_val_1d, proba_pos_val) # Используем вероятности

            # Сохраняем метрики фолда
            cv_metrics['accuracy'].append(acc)
            cv_metrics['precision'].append(prec)
            cv_metrics['recall'].append(rec)
            cv_metrics['f1'].append(f1)
            cv_metrics['roc_auc'].append(roc_auc)

            logging.info(f"Fold {fold + 1} Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}")

        except Exception as e:
            logging.error(f"Error during CV Fold {fold + 1}: {e}", exc_info=True)
            # Можно решить, прерывать ли CV или просто пропустить фолд


    # Расчет и логирование средних метрик по CV
    if cv_metrics['accuracy']: # Убедимся, что хотя бы 1 фолд успешен
        logging.info("--- Cross-Validation Summary ---")
        for metric_name, values in cv_metrics.items():
            mean_metric = np.mean(values)
            std_metric = np.std(values)
            logging.info(f"Average {metric_name.upper()}: {mean_metric:.4f} +/- {std_metric:.4f}")
    else:
        logging.warning("Cross-validation did not complete successfully for any fold.")


    # --- 4. Обучение Финальной Модели ---
    logging.info("Training final model on the full training set...")
    try:
        final_model = LogisticModel(
            learning_rate=config['model_params']['learning_rate'],
            iterations=config['model_params']['iterations'],
            lambda_reg=config['model_params']['lambda_reg'],
            verbose=config['model_params'].get('verbose', True) # Берем из конфига или True по умолч.
        )
        final_model.fit(X_train_processed, y_train)
        logging.info("Final model training complete.")
    except Exception as e:
         logging.error(f"Error during final model training: {e}", exc_info=True)
         return


    # --- 5. Оценка на Тестовой Выборке ---
    logging.info("Evaluating final model on the test set...")
    try:
        y_pred_proba_test = final_model.predict_proba(X_test_processed)
        y_pred_test = final_model.predict(X_test_processed)

        y_test_1d = y_test.ravel()
        y_pred_test_1d = y_pred_test.ravel()
        proba_pos_test = y_pred_proba_test[:, 1]

        test_acc = accuracy_score(y_test_1d, y_pred_test_1d)
        test_prec = precision_score(y_test_1d, y_pred_test_1d)
        test_rec = recall_score(y_test_1d, y_pred_test_1d)
        test_f1 = f1_score(y_test_1d, y_pred_test_1d)
        test_roc_auc = roc_auc_score_func(y_test_1d, proba_pos_test)

        logging.info("--- Test Set Evaluation ---")
        logging.info(f"Accuracy:  {test_acc:.4f}")
        logging.info(f"Precision: {test_prec:.4f}")
        logging.info(f"Recall:    {test_rec:.4f}")
        logging.info(f"F1 Score:  {test_f1:.4f}")
        logging.info(f"ROC AUC:   {test_roc_auc:.4f}")

    except Exception as e:
        logging.error(f"Error during test set evaluation: {e}", exc_info=True)
        # Продолжаем, чтобы сохранить модель и визуализации, если возможно


    # --- 6. Визуализация ---
    logging.info("Generating visualizations...")
    output_paths = config.get('output', {}) # Берем секцию output из конфига

    # Создаем директорию для output, если не существует
    output_dir = os.path.dirname(output_paths.get("roc_curve_path", ".")) # Берем путь из любого output файла
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")


    # 6.1 Кривая обучения (Loss vs Epoch)
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(final_model.history['loss']) + 1), final_model.history['loss'])
        plt.title('Кривая обучения (Loss vs Epoch)')
        plt.xlabel('Эпоха (Итерация)')
        plt.ylabel('Функция потерь (Cost)')
        plt.grid(True)
        loss_curve_path = output_paths.get("loss_curve_path", "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        logging.info(f"Learning curve saved to {loss_curve_path}")
    except Exception as e:
        logging.warning(f"Could not generate or save learning curve: {e}", exc_info=True)


    # 6.2 ROC-кривая
    try:
        fpr, tpr, _ = sk_roc_curve(y_test_1d, proba_pos_test)
        roc_auc_value = sk_auc(fpr, tpr) # Используем sklearn.metrics.auc

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc_value:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Линия случайного угадывания
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) кривая')
        plt.legend(loc="lower right")
        plt.grid(True)
        roc_curve_path = output_paths.get("roc_curve_path", "roc_curve.png")
        plt.savefig(roc_curve_path)
        plt.close()
        logging.info(f"ROC curve saved to {roc_curve_path}")
    except Exception as e:
        logging.warning(f"Could not generate or save ROC curve: {e}", exc_info=True)


    # 6.3 Матрица ошибок
    try:
        cm = sk_confusion_matrix(y_test_1d, y_pred_test_1d)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Матрица ошибок (Confusion Matrix)')
        plt.ylabel('Истинный класс')
        plt.xlabel('Предсказанный класс')
        conf_matrix_path = output_paths.get("confusion_matrix_path", "confusion_matrix.png")
        plt.savefig(conf_matrix_path)
        plt.close()
        logging.info(f"Confusion matrix saved to {conf_matrix_path}")
    except Exception as e:
        logging.warning(f"Could not generate or save confusion matrix: {e}", exc_info=True)


    # --- 7. Сохранение артефактов (модель и трансформер) ---
    logging.info("Saving trained model and data transformer...")

    # Сохранение трансформера
    transformer_path = output_paths.get("transformer_path", "data_transformer.pkl")
    try:
        with open(transformer_path, 'wb') as f:
            pickle.dump(transformer, f) # Используем pickle (или joblib)
        logging.info(f"Data transformer saved to {transformer_path}")
    except Exception as e:
        logging.error(f"Error saving data transformer: {e}", exc_info=True)

    # Сохранение модели
    model_path = output_paths.get("model_path", "credit_scoring_model.pkl")
    try:
        # Используем метод save_model из нашего класса LogisticModel
        # Передаем путь, куда сохранить веса
        final_model.save_model(model_path) # Метод внутри должен логировать успех
    except Exception as e:
        logging.error(f"Error saving the logistic model: {e}", exc_info=True)


    logging.info("Training process finished.")


# --- Точка входа ---
if __name__ == "__main__":
    # 1. Загружаем конфигурацию
    config = load_config('config.json') # Путь можно передавать через аргументы командной строки argparse

    # 2. Настраиваем логирование на основе конфига
    setup_logging(config.get('logging', {})) # Передаем секцию logging или пустой словарь

    # 3. Запускаем основной процесс обучения
    run_training(config)