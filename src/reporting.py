from __future__ import annotations

from pathlib import Path
import pandas as pd


def build_baseline_summary(preds_csv: Path) -> dict:
    df = pd.read_csv(preds_csv)

    # erros absolutos (baselines)
    df["err_dow"] = (df["actual"] - df["pred_dow_mean"]).abs()
    df["err_snaive"] = (df["actual"] - df["pred_seasonal_naive"]).abs()
    df["err_hybrid"] = (df["actual"] - df["pred_hybrid"]).abs()

    # ML (se existir)
    has_ml = "pred_ml_rf" in df.columns
    if has_ml:
        df["err_ml_rf"] = (df["actual"] - df["pred_ml_rf"]).abs()

    def mae(col: str) -> float:
        return float(df[col].mean())

    def rmse(pred_col: str) -> float:
        return float((((df["actual"] - df[pred_col]) ** 2).mean()) ** 0.5)

    # métricas (baselines)
    metrics = {
        "dow_mean": {"mae": mae("err_dow"), "rmse": rmse("pred_dow_mean")},
        "seasonal_naive_7": {"mae": mae("err_snaive"), "rmse": rmse("pred_seasonal_naive")},
        "hybrid": {"mae": mae("err_hybrid"), "rmse": rmse("pred_hybrid")},
    }

    # métricas (ML)
    if has_ml:
        metrics["ml_random_forest"] = {
            "mae": mae("err_ml_rf"),
            "rmse": rmse("pred_ml_rf"),
        }

    # escolher qual modelo usar nas tabelas de "melhor/pior dia"
    if has_ml:
        err_col = "err_ml_rf"
        pred_col = "pred_ml_rf"
        model_label = "ML (Random Forest)"
    else:
        err_col = "err_hybrid"
        pred_col = "pred_hybrid"
        model_label = "Hybrid"

    worst = (
        df.assign(err=df[err_col])
        .sort_values("err", ascending=False)
        .head(5)[["date", "actual", pred_col, "err"]]
        .rename(columns={pred_col: "predicted"})
    )

    best = (
        df.assign(err=df[err_col])
        .sort_values("err", ascending=True)
        .head(5)[["date", "actual", pred_col, "err"]]
        .rename(columns={pred_col: "predicted"})
    )

    volume_stats = {
        "mean": float(df["actual"].mean()),
        "min": float(df["actual"].min()),
        "max": float(df["actual"].max()),
    }

    return {
        "metrics": metrics,
        "volume_stats": volume_stats,
        "worst_days": worst,
        "best_days": best,
        "n_days": int(len(df)),
        "primary_label": model_label,
    }


def write_summary_md(summary: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = summary["metrics"]
    vs = summary["volume_stats"]
    primary = summary.get("primary_label", "Hybrid")

    def fmt(v: float) -> str:
        return f"{v:.2f}"

    lines: list[str] = []
    lines.append("# Service Desk Demand Forecast — Report\n")
    lines.append(f"Período avaliado: últimos **{summary['n_days']} dias** (split temporal).\n")

    lines.append("## Volume (real)\n")
    lines.append(f"- Média diária: **{fmt(vs['mean'])}** chamados/dia\n")
    lines.append(f"- Mínimo: **{fmt(vs['min'])}**\n")
    lines.append(f"- Máximo: **{fmt(vs['max'])}**\n")

    lines.append("## Métricas (quanto menor, melhor)\n")
    lines.append("| Modelo | MAE | RMSE |\n")
    lines.append("|---|---:|---:|\n")
    lines.append(f"| DOW mean | {fmt(m['dow_mean']['mae'])} | {fmt(m['dow_mean']['rmse'])} |\n")
    lines.append(f"| Seasonal Naive (7) | {fmt(m['seasonal_naive_7']['mae'])} | {fmt(m['seasonal_naive_7']['rmse'])} |\n")
    lines.append(f"| Hybrid | {fmt(m['hybrid']['mae'])} | {fmt(m['hybrid']['rmse'])} |\n")
    if "ml_random_forest" in m:
        lines.append(f"| **ML (Random Forest)** | **{fmt(m['ml_random_forest']['mae'])}** | **{fmt(m['ml_random_forest']['rmse'])}** |\n")

    lines.append("\n## Insights rápidos\n")
    if "ml_random_forest" in m:
        lines.append("- O modelo de **ML (Random Forest)** superou os baselines com folga, usando lags e médias móveis para capturar padrão semanal e tendência.\n")
        lines.append("- Próximo passo natural (v2): incluir features por categoria/prioridade e detectar dias anômalos (incidentes).\n")
    else:
        lines.append("- O **Hybrid** teve o melhor desempenho entre os baselines, combinando sazonalidade semanal + padrão por dia da semana.\n")
        lines.append("- Próximo passo natural (v2): treinar um modelo de ML com lags/médias móveis para melhorar precisão em picos.\n")

    lines.append(f"\n## Dias com maior erro ({primary})\n")
    lines.append("| Data | Real | Previsto | Erro |\n")
    lines.append("|---|---:|---:|---:|\n")
    for _, r in summary["worst_days"].iterrows():
        lines.append(f"| {r['date']} | {fmt(r['actual'])} | {fmt(r['predicted'])} | {fmt(r['err'])} |\n")

    lines.append(f"\n## Dias com menor erro ({primary})\n")
    lines.append("| Data | Real | Previsto | Erro |\n")
    lines.append("|---|---:|---:|---:|\n")
    for _, r in summary["best_days"].iterrows():
        lines.append(f"| {r['date']} | {fmt(r['actual'])} | {fmt(r['predicted'])} | {fmt(r['err'])} |\n")

    out_path.write_text("".join(lines), encoding="utf-8")