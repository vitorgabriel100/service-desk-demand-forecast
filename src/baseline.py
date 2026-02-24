from __future__ import annotations

import pandas as pd


def temporal_train_test_split(df_daily: pd.DataFrame, test_days: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal (sem shuffle).
    Espera colunas: date (datetime), tickets (int/float)
    """
    df = df_daily.sort_values("date").reset_index(drop=True)
    if len(df) <= test_days + 14:
        raise ValueError("Série muito curta para split. Use mais dados ou reduza test_days.")

    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["date"].dt.dayofweek  # 0=Mon..6=Sun
    return out


def fit_dow_mean(train: pd.DataFrame) -> dict[int, float]:
    """
    Retorna um dicionário {dow: mean_tickets}
    """
    t = add_time_features(train)
    dow_mean = t.groupby("dow")["tickets"].mean().to_dict()
    return {int(k): float(v) for k, v in dow_mean.items()}


def predict_dow_mean(test: pd.DataFrame, dow_mean: dict[int, float]) -> pd.Series:
    t = add_time_features(test)
    # fallback: média global caso algum dow não exista no treino
    global_mean = sum(dow_mean.values()) / max(1, len(dow_mean))
    return t["dow"].map(dow_mean).fillna(global_mean).astype(float)


def predict_seasonal_naive(df_daily: pd.DataFrame, lag_days: int = 7) -> pd.Series:
    """
    Seasonal naive: previsão(t) = valor(t - lag_days)
    Retorna uma série alinhada com df_daily (mesmo index).
    """
    df = df_daily.sort_values("date").reset_index(drop=True)
    pred = df["tickets"].shift(lag_days)
    return pred.astype(float)


def predict_hybrid(test: pd.DataFrame, pred_dow: pd.Series, pred_snaive: pd.Series, alpha: float = 0.6) -> pd.Series:
    """
    Híbrido: alpha*dow + (1-alpha)*seasonal_naive
    alpha=0.6 tende a estabilizar bem.
    """
    # alinhamento por index
    p1 = pred_dow.reset_index(drop=True)
    p2 = pred_snaive.reset_index(drop=True)
    return (alpha * p1 + (1 - alpha) * p2).astype(float)