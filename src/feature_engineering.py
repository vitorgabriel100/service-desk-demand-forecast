from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_daily_series(raw_csv_path: Path) -> pd.DataFrame:
    """
    Lê tickets raw (nível ticket) e agrega para série diária:
    columns: date, tickets
    """
    df = pd.read_csv(raw_csv_path)

    # created_at -> datetime
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).copy()

    df["date"] = df["created_at"].dt.date
    daily = (
        df.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "tickets"})
    )

    # garante ordenação temporal
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily