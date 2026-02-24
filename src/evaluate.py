from __future__ import annotations

import math
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.astype(float).to_numpy()
    y_pred = y_pred.astype(float).to_numpy()
    return float((abs(y_true - y_pred)).mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = y_true.astype(float).to_numpy()
    y_pred = y_pred.astype(float).to_numpy()
    return float(math.sqrt(((y_true - y_pred) ** 2).mean()))


def make_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }