# app/app.py

import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from src.data import fetch_intraday_data, split_train_test
from src.features import build_feature_matrix
from src.models import scale_features, WINDOW_SIZE
from src.ensemble import StackedEnsemble

st.set_page_config(layout="wide", page_title="Alpha Backtester", page_icon="📈")
st.title("📈 Intraday Alpha Backtesting Framework")

# --- SIDEBAR ---
st.sidebar.title("Developed by Sanskar Jadhav")
ticker = st.sidebar.selectbox(
    "Equity Universe", 
    ["BTC-USD", "AAPL", "NVDA", "TSLA", "MSFT", "AMD", "META", "GOOGL", "AMZN", "NFLX", "SPY", "QQQ"]
)
st.sidebar.markdown("---")
st.sidebar.subheader("Backtest Parameters")

cost_per_trade = st.sidebar.slider(
    "Transaction Friction (%)", 
    0.0, 0.1, 0.02, 0.01,
    help="Simulates commission, slippage, and spread crossing costs."
) / 100 

intercept_neutral = st.sidebar.checkbox(
    "Intercept Neutralization", 
    value=False, 
    help="Demeans the prediction vector to ensure signal symmetry regardless of the session-level drift."
)

min_conviction = st.sidebar.slider(
    "Signal Conviction Threshold (%)", 
    20, 90, 30, 5,
    help="Filters for high-conviction alpha signals by ignoring predictions within the noise floor."
) / 100

# --- STABLE TRAINING ENGINE ---
@st.cache_resource(show_spinner=False)
def run_model_pipeline(ticker):
    raw_df = fetch_intraday_data(ticker)
    X, y = build_feature_matrix(raw_df)
    
    train_raw, test_raw = split_train_test(raw_df)
    train_idx, test_idx = X.index.isin(train_raw.index), X.index.isin(test_raw.index)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    X_tr_s, X_te_s, _ = scale_features(X_train.values, X_test.values)
    model = StackedEnsemble()
    model.fit(X_tr_s, y_train.values)
    
    pred_rets = model.predict(X_te_s)
    return raw_df, X_test, y_test, pred_rets

try:
    with st.spinner(f"Running Alpha Pipeline for {ticker}..."):
        raw_df, X_test, y_test, pred_rets = run_model_pipeline(ticker)

    # --- ALPHA SIGNAL LOGIC ---
    actual_rets = y_test.values[WINDOW_SIZE:]
    times = X_test.index[WINDOW_SIZE:]
    
    bias = np.mean(pred_rets) if intercept_neutral else 0
    alpha_forecast = pred_rets - bias
    
    std_dev = np.std(alpha_forecast)
    confidence_scores = np.clip(np.abs(alpha_forecast) / (std_dev * 2), 0.1, 1.0)
    
    raw_signals = np.sign(alpha_forecast)
    active_signals = np.where(confidence_scores >= min_conviction, raw_signals, 0)
    
    # BACKTEST MATH
    gross_rets = active_signals * actual_rets
    net_rets = gross_rets - (np.abs(active_signals) * cost_per_trade) 
    
    # --- PERFORMANCE METRICS ---
    trades_taken = np.count_nonzero(active_signals)
    hit_ratio = np.mean(np.sign(actual_rets[active_signals != 0]) == np.sign(pred_rets[active_signals != 0])) if trades_taken > 0 else 0
    
    cum_returns = np.exp(np.cumsum(net_rets)) - 1
    total_net_return = cum_returns[-1] * 100

    daily_std = np.std(net_rets) * np.sqrt(78)
    sharpe = (np.mean(net_rets) * 78 * 252) / (daily_std * np.sqrt(252)) if daily_std != 0 else 0
    
    peak = np.maximum.accumulate(cum_returns + 1)
    drawdown = (cum_returns + 1) / peak - 1
    max_dd = np.min(drawdown) * 100

    # --- METRIC DASHBOARD ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Information Hit Ratio", f"{hit_ratio:.2%}")
    prof_color = "normal" if total_net_return >= 0 else "inverse"
    m2.metric("Cumulative Alpha (Net)", f"{total_net_return:.2f}%", delta=f"{total_net_return:.2f}%", delta_color=prof_color)
    m3.metric("Sharpe Ratio", f"{sharpe:.2f}", help="Institutional grade > 2.0")
    m4.metric("Max Drawdown", f"{max_dd:.2f}%")

    # --- VISUALIZATION ---
    actual_prices = raw_df.loc[times, 'Close'].values
    pred_prices_raw = actual_prices * (1 + pred_rets)
    forecast_times = times + pd.Timedelta(minutes=5)
    backtest_date = times[0].strftime('%Y-%m-%d')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.8, 0.2])

    fig.add_trace(go.Scatter(x=times, y=actual_prices, name="Spot Price", 
                             line=dict(color='#00ff88', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=forecast_times, y=pred_prices_raw, name="Alpha Forecast (T+1)", 
                             line=dict(color='#ff00ff', dash='dot', width=1.5)), row=1, col=1)

    # Conviction Heatmap
    bar_colors = [f'rgba({"0, 255, 136" if s > 0 else "255, 75, 75"}, {c})' if s != 0 else 'rgba(100, 100, 100, 0.1)' 
                  for s, c in zip(active_signals, confidence_scores)]
    
    signal_labels = ["LONG" if s > 0 else "SHORT" if s < 0 else "NO TRADE" for s in active_signals]
    
    fig.add_trace(go.Bar(x=times, y=[1]*len(active_signals), marker_color=bar_colors,
                         name="Signal State", showlegend=False,
                         text=signal_labels,
                         hovertemplate="State: %{text}<br>Conviction: %{customdata:.1%}<extra></extra>",
                         customdata=confidence_scores), row=2, col=1)

    fig.update_layout(title=f"Alpha Analytics: {ticker} | Simulation Date: {backtest_date}", 
                      template="plotly_dark", height=600, hovermode="x unified", margin=dict(t=50, b=50))

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Signal (Long/Short)", showticklabels=False, row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

    # EQUITY CURVE
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=times, y=cum_returns * 100, fill='tozeroy', 
                                    name="Backtested Equity", line=dict(color='#3399ff', width=3)))
    fig_equity.update_layout(title="Net Performance Curve (%)", 
                             xaxis_title="Simulation Time (EST)", yaxis_title="Cumulative PnL (%)",
                             template="plotly_dark", height=400)
    st.plotly_chart(fig_equity, use_container_width=True)

except Exception as e:
    st.error(f"Backtest Runtime Error: {e}")
