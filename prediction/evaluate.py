"""Evaluation helpers for rider stress prediction."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LEVEL_THRESHOLDS = [4,8]  # low <4, mid <8, else high


def regression_metrics(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true,y_pred)),
        'RMSE': float(mean_squared_error(y_true,y_pred)**0.5),
        'R2': float(r2_score(y_true,y_pred))
    }

def classify_level(score: float) -> str:
    if score < LEVEL_THRESHOLDS[0]:
        return '低压力'
    if score < LEVEL_THRESHOLDS[1]:
        return '中压力'
    return '高压力'

def classification_metrics(y_true, y_pred):
    true_levels = [classify_level(s) for s in y_true]
    pred_levels = [classify_level(s) for s in y_pred]
    acc = np.mean([t==p for t,p in zip(true_levels,pred_levels)])
    return {'Level_Accuracy': float(acc)}
