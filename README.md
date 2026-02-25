# Intraday Alpha Forecasting and Execution Framework

## A Stacked Deep Learning Approach
------------------------------------------------------------------------

## Abstract

Financial time-series data at the intraday level (e.g., 5-minute
intervals) is notoriously non-stationary and exhibits a remarkably low
Signal-to-Noise Ratio (SNR).

This project implements a quantitative research and backtesting
framework designed to extract idiosyncratic alpha from liquid equity
markets. By leveraging a **Stacked Ensemble Architecture**---combining
the non-linear pattern recognition capabilities of TensorFlow/Keras Deep
Neural Networks with the structural stability of regularized linear
models---this framework generates, filters, and backtests T+1
directional forecasts under realistic execution constraints.

------------------------------------------------------------------------

# 1. Machine Learning Architecture

The predictive engine relies on a hierarchical stacking methodology
(Stacked Generalization) to mitigate the variance inherent in financial
forecasting.

## 1.1 Data Preprocessing & Stationarity

Raw price sequences are non-stationary. To satisfy weak stationarity
assumptions, prices are transformed into logarithmic returns:

$r_t = ln(P_t / P\_{t-1})$

Features are subsequently standardized using strict temporal
separation.\
Z-score normalization is fitted exclusively on the expanding training
window to prevent look-ahead bias (data leakage).

## 1.2 Base Learners (L0)

### Deep Neural Network (TensorFlow 2.x / Keras)

A Multi-Layer Perceptron (MLP) designed to map complex, non-linear
feature interactions.

Regularization techniques include: 
- Dropout layers
- L2 (Ridge) weight regularization

### Regularized Linear Models (Scikit-Learn)

Algorithms such as Ridge Regression act as high-bias, low-variance
anchors, effectively handling multicollinearity in engineered momentum
and volatility features.

## 1.3 Meta-Learner (L1)

The secondary layer ingests cross-validated predictions from base
learners and computes an optimal weighted synthesis, reducing
generalization error and producing the final raw alpha forecast.

------------------------------------------------------------------------

# 2. Financial Engineering & Signal Processing

## 2.1 Intercept Neutralization (Session-Relative Bias)

$A_t = y_hat_t - mean(y_hat)$

This isolates idiosyncratic alpha by removing session-level drift.

## 2.2 Conviction-Weighted Thresholding

$Z_t = \|A_t\| / sigma_A$

$If Z_t \< tau$ → signal suppressed $(S_t = 0).$\
Otherwise → trade executed.

------------------------------------------------------------------------

# 3. Friction-Adjusted Backtesting

$Let: - S_t ∈ {-1, 0, 1} = execution signal$\
- c = transaction cost per trade

$R_net,t = (S_t \* r_actual,t) - (c \* \|S_t\|)$

------------------------------------------------------------------------

## 3.1 Performance Metrics

### Information Hit Ratio

Directional accuracy of active signals.

### Annualized Sharpe Ratio

$Sharpe = sqrt(252 \* 72) \* mu_net / sigma_net$

### Maximum Drawdown (MDD)

$MDD = min_t (E_t / max\_{tau ≤ t} E_tau - 1)$

------------------------------------------------------------------------

# Tech Stack

**Deep Learning** 
- TensorFlow 2.x
- Keras

**Machine Learning / Math** 
- Scikit-Learn
- NumPy
- SciPy

**Data Pipeline** 
- Pandas
- yfinance

**Visualization & UI** 
- Plotly
- Streamlit

------------------------------------------------------------------------

# Repository Structure

    ├── app.py
    ├── src/
    |   ├── config.py
    │   ├── data.py
    │   ├── ensemble.py
    │   ├── features.py
    │   ├── models.py
    │   ├── regime.py
    │   ├── risk.py
    │   └── validation.py
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

# Usage

``` bash
pip install -r requirements.txt
streamlit run app/app.py
```
Or can visit online: [Alpha Backtest](https://alphabacktest.streamlit.app/).
