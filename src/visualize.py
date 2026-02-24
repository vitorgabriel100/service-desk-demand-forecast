from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates


def plot_forecast(df_plot: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = df_plot.copy()
    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(df["date"], df["actual"], label="actual")
    ax.plot(df["date"], df["predicted"], label="predicted")

    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("tickets")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))        # 1 label a cada 4 dias
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))

    # ✅ Rotação e espaçamento
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
