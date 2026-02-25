# Intraday Alpha Backtesting Framework
A high-frequency signal research and backtesting engine designed to evaluate idiosyncratic alpha in liquid equity universes. This framework leverages an Ensemble Machine Learning approach to forecast T+1 returns while accounting for institutional-grade execution constraints.

## Quant Research Highlights
Ensemble Alpha Signal: Utilizes a stacked ensemble architecture to capture non-linear price patterns across 5-minute intervals.

Intercept Neutralization: Implements a session-level de-meaning process to neutralize the global drift (systematic bias). This allows the framework to extract tradable signals even during one-way trending markets.

Confidence-Weighted Execution: Signals are not binary; they are scaled by the model's conviction (Z-score of the forecast distribution). Only signals exceeding the Noise Floor (Conviction Threshold) trigger a capital allocation.

Friction-Adjusted Backtesting: Real-world simulation including a Transaction Friction parameter to account for the bid-ask spread, slippage, and commissions.

## Methodology & Metrics
The framework evaluates the "quality of intelligence" using institutional performance attribution:

Information Hit Ratio: The pure predictive accuracy of the alpha signal before transaction costs are applied.

Sharpe Ratio (Annualized): Measures the risk-adjusted return, normalizing for the volatility of the intraday strategy.

Maximum Drawdown: Tracks the largest peak-to-trough decline, representing the strategy's historical risk profile.

Capacity/Participation: Monitors the strategy's utilization rate, ensuring the alpha is not confined to a statistically insignificant number of bars.

## Tech Stack
Language: Python 3.x

Modeling: Scikit-Learn (Ensemble methods), NumPy (Vectorized Backtesting)

Visualization: Plotly (Dynamic Alpha Analytics)

Interface: Streamlit

📂 Project Structure
Plaintext
├── app/
│   └── app.py           # Dashboard & Signal Visualization
├── src/
│   ├── data.py          # Ingestion logic & API integration
│   ├── features.py      # Feature engineering (Stationarity & Scaling)
│   ├── models.py        # Model architecture & Training pipeline
│   └── ensemble.py      # Stacked Regressor implementation
└── requirements.txt

## Technical Concepts Explored

Intercept Neutralization (The "Intraday Focus")
In many intraday scenarios, a model may suffer from Label Bias if the training data has a strong trend. This framework implements Intercept Neutralization to center the prediction vector around zero. This ensures the strategy focuses on mean-reversion or local momentum rather than simply betting on the day's trend, which is a common pitfall in retail trading bots.

Conviction-Weighted Allocation
Instead of fixed-size positions, this framework simulates a dynamic sizing approach. The signal opacity in the visualization represents the Signal Strength. High-conviction signals are prioritized, while low-conviction signals are filtered out as market noise, significantly improving the strategy's Profit Factor.
