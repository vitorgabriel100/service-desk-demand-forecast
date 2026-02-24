# Service Desk Demand Forecast — Report
Período avaliado: últimos **28 dias** (split temporal).
## Volume (real)
- Média diária: **86.14** chamados/dia
- Mínimo: **21.00**
- Máximo: **168.00**
## Métricas (quanto menor, melhor)
| Modelo | MAE | RMSE |
|---|---:|---:|
| DOW mean | 23.10 | 28.05 |
| Seasonal Naive (7) | 28.43 | 36.72 |
| Hybrid | 21.42 | 28.81 |
| **ML (Random Forest)** | **4.48** | **6.44** |

## Insights rápidos
- O modelo de **ML (Random Forest)** superou os baselines com folga, usando lags e médias móveis para capturar padrão semanal e tendência.
- Próximo passo natural (v2): incluir features por categoria/prioridade e detectar dias anômalos (incidentes).

## Dias com maior erro (ML (Random Forest))
| Data | Real | Previsto | Erro |
|---|---:|---:|---:|
| 2026-02-09 | 145.00 | 127.74 | 17.26 |
| 2026-01-19 | 168.00 | 151.29 | 16.71 |
| 2026-01-26 | 134.00 | 120.47 | 13.53 |
| 2026-02-03 | 86.00 | 95.95 | 9.95 |
| 2026-01-27 | 48.00 | 55.43 | 7.43 |

## Dias com menor erro (ML (Random Forest))
| Data | Real | Previsto | Erro |
|---|---:|---:|---:|
| 2026-01-24 | 46.00 | 45.73 | 0.27 |
| 2026-01-31 | 64.00 | 63.63 | 0.37 |
| 2026-02-13 | 51.00 | 51.51 | 0.51 |
| 2026-02-05 | 130.00 | 129.28 | 0.72 |
| 2026-01-29 | 104.00 | 104.88 | 0.88 |
