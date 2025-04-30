# credit_scoring/metrics/__init__.py

# Импортируем нужные функции из модуля metrics.py (который лежит в этой же папке)
# чтобы они были доступны напрямую из пакета credit_scoring.metrics
from .metrics import (
    accuracy_score,
    confusion_matrix_vals,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score_func,
    get_roc_curve_data
)

# Можно добавить __all__, чтобы явно указать, что экспортируется (хорошая практика)
__all__ = [
    'accuracy_score',
    'confusion_matrix_vals',
    'precision_score',
    'recall_score',
    'f1_score',
    'roc_auc_score_func',
    'get_roc_curve_data'
]