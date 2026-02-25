📊 Service Desk Demand Forecast

Previsão de volume de chamados com Machine Learning para apoio à decisão operacional

📌 Contexto

Atuo como Analista de Service Desk Júnior e desenvolvi este projeto para analisar padrões de chamados e prever demanda diária, apoiando decisões como dimensionamento de equipe, identificação de períodos críticos e planejamento operacional.

O foco do projeto é IA aplicada, comparando baselines clássicos de séries temporais com um modelo de Machine Learning, utilizando validação temporal adequada.

🎯 Objetivo

Prever o volume diário de chamados

Comparar baselines tradicionais vs modelo de ML

Avaliar impacto prático na redução do erro de previsão

Gerar insights acionáveis para operação de Service Desk

🧠 Metodologia

1️⃣ Dados

Dataset simulado e realista de chamados de Service Desk

Agregação para série temporal diária

Validação com split temporal (treino no passado, teste no futuro)

2️⃣ Baselines

Day-of-Week Mean (média por dia da semana)

Seasonal Naive (7) – repete o valor de 7 dias atrás

Hybrid – combinação dos dois baselines

3️⃣ Machine Learning

Modelo: Random Forest Regressor

Features utilizadas:

Lags de demanda (1 e 7 dias)

Médias móveis (7 e 14 dias)

Tendência semanal

Dia da semana (one-hot encoding)

📈 Resultados

Período avaliado: últimos 28 dias

Modelo	MAE	RMSE
DOW mean	23.10	28.05
Seasonal Naive (7)	28.43	36.72
Hybrid	21.42	28.81
ML (Random Forest)	4.48	6.44

📉 O modelo de Machine Learning reduziu drasticamente o erro em relação aos baselines, demonstrando sua eficácia para previsão operacional.

🔍 Insights Operacionais

O Random Forest capturou com eficiência sazonalidade semanal e tendência de curto prazo

Maiores erros ocorrem em dias de pico anormal, típicos de incidentes

O modelo é especialmente útil para planejamento de escala e antecipação de sobrecarga

📊 Visualizações

Baseline: outputs/charts/baseline_real_vs_pred.png

ML: outputs/charts/ml_real_vs_pred.png

🛠️ Estrutura do Projeto
service-desk-demand-forecast/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── load_data.py
│   ├── synthetic_data.py
│   ├── feature_engineering.py
│   ├── baseline.py
│   ├── model.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── reporting.py
│
├── outputs/
│   ├── charts/
│   ├── reports/
│   └── model/
│
├── main.py
├── requirements.txt
└── README.md

▶️ Como executar (Windows)

python -m venv .venv

.venv\Scripts\activate

pip install -r requirements.txt


python main.py --make-data

python main.py --train

python main.py --report


🚀 Próximos Passos (v2)

Inclusão de features por categoria, prioridade e fila

Detecção automática de dias anômalos (incidentes)

Exposição do modelo via API (FastAPI) para consumo operacional

Simulação de cenários de escala (what-if)

🧩 Tecnologias

Python

Pandas, NumPy

Scikit-learn

Matplotlib

Machine Learning aplicado a séries temporais

🏁 Conclusão

Este projeto demonstra como Machine Learning pode ser aplicado de forma prática para antecipar demanda, reduzir incertezas operacionais e apoiar decisões em ambientes reais de Service Desk.
