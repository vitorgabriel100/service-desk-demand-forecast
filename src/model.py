from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


@dataclass
class ModelArtifacts:
    model_path: Path
    metadata_path: Path


def make_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Features simples e eficazes para demanda diária.
    Espera: date (datetime), tickets (num)
    Retorna df com colunas de features + target.
    """
    df = df_daily.sort_values("date").reset_index(drop=True).copy()
    df["dow"] = df["date"].dt.dayofweek

    # Lags (demanda recente)
    df["lag_1"] = df["tickets"].shift(1)
    df["lag_7"] = df["tickets"].shift(7)

    # Médias móveis (tendência/nível)
    df["roll_7"] = df["tickets"].rolling(7).mean()
    df["roll_14"] = df["tickets"].rolling(14).mean()

    # Tendência simples
    df["trend_7"] = df["tickets"] - df["tickets"].shift(7)

    return df


def temporal_train_test(df_feats: pd.DataFrame, test_days: int = 28) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_feats.sort_values("date").reset_index(drop=True)
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test


def train_random_forest(df_daily: pd.DataFrame, test_days: int = 28, seed: int = 42) -> tuple[Pipeline, dict]:
    """
    Treina um RandomForestRegressor com pipeline (API-ready).
    Retorna: pipeline treinado, metadata (métricas e features)
    """
    feats = make_features(df_daily)

    # remove linhas iniciais com NaN (por causa de lag/rolling)
    feats = feats.dropna(subset=["lag_7", "roll_14"]).reset_index(drop=True)

    train, test = temporal_train_test(feats, test_days=test_days)

    feature_cols_num = ["lag_1", "lag_7", "roll_7", "roll_14", "trend_7"]
    feature_cols_cat = ["dow"]

    X_train = train[feature_cols_num + feature_cols_cat]
    y_train = train["tickets"].astype(float)

    X_test = test[feature_cols_num + feature_cols_cat]
    y_test = test["tickets"].astype(float)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    # preds
    pred = pipe.predict(X_test)

    metadata = {
        "model_type": "RandomForestRegressor",
        "test_days": int(test_days),
        "seed": int(seed),
        "feature_cols_num": feature_cols_num,
        "feature_cols_cat": feature_cols_cat,
        "y_test": y_test.tolist(),
        "pred": pred.tolist(),
        "test_dates": test["date"].dt.strftime("%Y-%m-%d").tolist(),
    }

    return pipe, metadata


def save_artifacts(pipe: Pipeline, metadata: dict, outputs_dir: Path) -> ModelArtifacts:
    model_dir = outputs_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "metadata.json"

    joblib.dump(pipe, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return ModelArtifacts(model_path=model_path, metadata_path=meta_path)