from __future__ import annotations

import argparse
from pathlib import Path

from src.load_data import ensure_dirs, write_readme_seed_hint
from src.synthetic_data import generate_synthetic_tickets_csv
from src.feature_engineering import build_daily_series


BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs"


def cmd_make_data() -> None:
    ensure_dirs(BASE_DIR)

    raw_path = DATA_RAW / "tickets_raw.csv"
    generate_synthetic_tickets_csv(
        out_path=raw_path,
        start_date="2025-01-01",
        end_date="2026-02-15",
        seed=42,
        daily_min=40,
        daily_max=140,
    )

    # Build daily aggregation for modeling
    processed_daily_path = DATA_PROCESSED / "tickets_daily.csv"
    df_daily = build_daily_series(raw_csv_path=raw_path)
    df_daily.to_csv(processed_daily_path, index=False)

    write_readme_seed_hint()
    print(f"[OK] Dataset gerado: {raw_path}")
    print(f"[OK] Série diária gerada: {processed_daily_path}")


def cmd_train() -> None:
    ensure_dirs(BASE_DIR)

    import pandas as pd
    from src.baseline import (
        temporal_train_test_split,
        fit_dow_mean,
        predict_dow_mean,
        predict_seasonal_naive,
        predict_hybrid,
    )
    from src.evaluate import make_metrics
    from src.visualize import plot_forecast

    daily_path = DATA_PROCESSED / "tickets_daily.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"Não encontrei {daily_path}. Rode antes: python main.py --make-data")

    df = pd.read_csv(daily_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train, test = temporal_train_test_split(df, test_days=28)

    # Baseline DOW mean
    dow_mean = fit_dow_mean(train)
    pred_dow = predict_dow_mean(test, dow_mean)

    # Seasonal naive (precisa de valores 7 dias atrás)
    pred_snaive_full = predict_seasonal_naive(df, lag_days=7)
    pred_snaive = pred_snaive_full.iloc[-len(test):].reset_index(drop=True)

    # Se tiver NaN no início do recorte (caso série curta), preenche com DOW
    pred_snaive = pred_snaive.fillna(pred_dow.reset_index(drop=True))

    # Híbrido (recomendado)
    pred_hybrid = predict_hybrid(test, pred_dow, pred_snaive, alpha=0.6)

    # Avaliação
    y_true = test["tickets"]
    m_dow = make_metrics(y_true, pred_dow)
    m_snaive = make_metrics(y_true, pred_snaive)
    m_hybrid = make_metrics(y_true, pred_hybrid)

    # Salvar previsões (para report depois / futuro API)
    out_preds = OUTPUTS / "reports" / "baseline_predictions.csv"
    df_out = pd.DataFrame({
        "date": test["date"],
        "actual": y_true.astype(float).values,
        "pred_dow_mean": pred_dow.values,
        "pred_seasonal_naive": pred_snaive.values,
        "pred_hybrid": pred_hybrid.values,
    })
    df_out.to_csv(out_preds, index=False)

    # Gráfico do melhor baseline (híbrido)
    chart_path = OUTPUTS / "charts" / "baseline_real_vs_pred.png"
    df_plot = pd.DataFrame({
        "date": test["date"],
        "actual": y_true.astype(float).values,
        "predicted": pred_hybrid.values,
        "model_name": ["baseline_hybrid"] * len(test),
    })
    plot_forecast(df_plot, chart_path, title="Baseline (Hybrid) — Real vs Previsto")
        # ===== ML Model (Random Forest) =====
    from src.model import train_random_forest, save_artifacts
    from src.evaluate import make_metrics

    pipe, meta = train_random_forest(df, test_days=28, seed=42)

    y_test = pd.Series(meta["y_test"])
    pred_ml = pd.Series(meta["pred"])
    m_ml = make_metrics(y_test, pred_ml)

    # salva artefatos (API-ready)
    arts = save_artifacts(pipe, meta, OUTPUTS)

    # salva previsões ML no mesmo CSV pra report/README
    df_out["pred_ml_rf"] = pred_ml.values
    out_preds = OUTPUTS / "reports" / "baseline_predictions.csv"
    df_out.to_csv(out_preds, index=False)

    # gráfico ML
    chart_ml = OUTPUTS / "charts" / "ml_real_vs_pred.png"
    df_plot_ml = pd.DataFrame({
        "date": test["date"],
        "actual": y_true.astype(float).values,
        "predicted": pred_ml.values,
        "model_name": ["ml_random_forest"] * len(test),
    })
    plot_forecast(df_plot_ml, chart_ml, title="ML (Random Forest) — Real vs Previsto")

    print(f" - ML (RF)    | MAE={m_ml['mae']:.2f} | RMSE={m_ml['rmse']:.2f}")
    print(f"[OK] Modelo salvo em: {arts.model_path}")
    print(f"[OK] Metadata salva em: {arts.metadata_path}")
    print(f"[OK] Gráfico ML salvo em: {chart_ml}")
    # Console summary
    print("[OK] Baselines avaliados (últimos 28 dias):")
    print(f" - DOW mean   | MAE={m_dow['mae']:.2f} | RMSE={m_dow['rmse']:.2f}")
    print(f" - SNaive(7)  | MAE={m_snaive['mae']:.2f} | RMSE={m_snaive['rmse']:.2f}")
    print(f" - Hybrid     | MAE={m_hybrid['mae']:.2f} | RMSE={m_hybrid['rmse']:.2f}")
    print(f"[OK] Previsões salvas em: {out_preds}")
    print(f"[OK] Gráfico salvo em: {chart_path}")


def cmd_report() -> None:
    ensure_dirs(BASE_DIR)

    from src.reporting import build_baseline_summary, write_summary_md

    preds_csv = OUTPUTS / "reports" / "baseline_predictions.csv"
    if not preds_csv.exists():
        raise FileNotFoundError(f"Não encontrei {preds_csv}. Rode antes: python main.py --train")

    summary = build_baseline_summary(preds_csv)
    out_md = OUTPUTS / "reports" / "summary.md"
    write_summary_md(summary, out_md)

    print(f"[OK] Relatório gerado em: {out_md}")


def cmd_all() -> None:
    cmd_make_data()
    cmd_train()
    cmd_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Service Desk Demand Forecast (CLI)",
    )
    parser.add_argument("--make-data", action="store_true", help="Gera dataset simulado e série diária.")
    parser.add_argument("--train", action="store_true", help="Treina baseline/modelo e salva artefatos.")
    parser.add_argument("--report", action="store_true", help="Gera gráficos e relatório.")
    parser.add_argument("--all", action="store_true", help="Roda pipeline completo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not any([args.make_data, args.train, args.report, args.all]):
        print("Nenhuma opção informada. Use: --make-data, --train, --report ou --all")
        return

    if args.all:
        cmd_all()
        return

    if args.make_data:
        cmd_make_data()
    if args.train:
        cmd_train()
    if args.report:
        cmd_report()


if __name__ == "__main__":
    main()