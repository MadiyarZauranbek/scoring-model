# credit_scoring/metrics/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix as sk_confusion_matrix

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Расчет точности (accuracy)."""
    return np.mean(y_true == y_pred)

def confusion_matrix_vals(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1):
    """Возвращает TP, FP, FN, TN."""
    tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred).ravel()
    # # Ручная реализация, если нужно:
    # tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    # fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    # fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    # tn = np.sum((y_true != pos_label) & (y_pred != pos_label))
    return tp, fp, fn, tn

def precision_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1) -> float:
    """Расчет точности (precision)."""
    tp, fp, _, _ = confusion_matrix_vals(y_true, y_pred, pos_label)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1) -> float:
    """Расчет полноты (recall)."""
    tp, _, fn, _ = confusion_matrix_vals(y_true, y_pred, pos_label)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label=1) -> float:
    """Расчет F1-меры."""
    prec = precision_score(y_true, y_pred, pos_label)
    rec = recall_score(y_true, y_pred, pos_label)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def roc_auc_score_func(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Расчет ROC AUC. Требует вероятности положительного класса.
    y_pred_proba - это массив вероятностей для класса 1 (положительного).
    """
    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
         # Если predict_proba вернул 2 колонки, берем вторую
        y_pred_proba = y_pred_proba[:, 1]
    elif y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 2:
        raise ValueError("roc_auc_score_func ожидает вероятности для бинарной классификации")
    # Если y_pred_proba уже одномерный, предполагаем, что это вероятности класса 1

    return roc_auc_score(y_true, y_pred_proba)

# Функция для получения данных для ROC-кривой (используем sklearn)
def get_roc_curve_data(y_true: np.ndarray, y_pred_proba: np.ndarray):
     """Возвращает FPR, TPR и пороги для построения ROC кривой."""
     if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[:, 1]
     fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
     return fpr, tpr, thresholds