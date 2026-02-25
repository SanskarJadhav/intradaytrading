# app/app.py

import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
from src.data import fetch_intraday_data, split_train_test
from src.features import build_feature_matrix
from src.models import scale_features, WINDOW_SIZE
from src.ensemble import StackedEnsemble
from src.regime import RegimeDetector 

st.set_page_config(layout="wide", page_title="Alpha Backtester", page_icon="📈")
st.title("📈 Intraday Alpha Backtesting Framework")

# --- CACHE KEY LOGIC ---
def get_session_key():
    """Generates a unique key based on the current New York date to force cache refresh daily."""
    ny_tz = pytz.timezone('America/New_York')
    return datetime.now(ny_tz).strftime("%Y-%m-%d")

# --- SIDEBAR ---
st.sidebar.title("Developed by Sanskar Jadhav")
ticker = st.sidebar.selectbox(
    "Equity Universe", 
    ["SPY", "QQQ", "AAPL", "NVDA", "TSLA", "MSFT", "AMD", "META", "GOOGL", "AMZN", "NFLX", "BTC-USD"]
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
    value=True, 
    help="Demeans the prediction vector to ensure signal symmetry regardless of the session-level drift."
)

min_conviction = st.sidebar.slider(
    "Signal Conviction Threshold (%)", 
    20, 95, 80, 5,
    help="Filters for high-conviction alpha signals by ignoring predictions within the noise floor."
) / 100

# --- TRAINING ENGINE ---
@st.cache_resource(show_spinner=False)
def run_model_pipeline(ticker, session_key):
    raw_df = fetch_intraday_data(ticker)
    X, y = build_feature_matrix(raw_df)
    
    train_raw, test_raw = split_train_test(raw_df)
    train_idx, test_idx = X.index.isin(train_raw.index), X.index.isin(test_raw.index)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    X_tr_s, X_te_s, _ = scale_features(X_train.values, X_test.values)
    
    # Regime Detection
    rd = RegimeDetector()
    rd.fit_predict(X_train) 
    test_regimes = rd.predict(X_test)
    
    model = StackedEnsemble()
    model.fit(X_tr_s, y_train.values)
    
    pred_rets = model.predict(X_te_s)
    return raw_df, X_test, y_test, pred_rets, test_regimes

try:
    current_session = get_session_key()
    with st.spinner(f"Running Alpha Pipeline for {ticker}..."):
        raw_df, X_test, y_test, pred_rets, test_regimes = run_model_pipeline(ticker, current_session)

    # --- ALPHA SIGNAL LOGIC ---
    actual_rets = y_test.values[WINDOW_SIZE:]
    times = X_test.index[WINDOW_SIZE:]
    regimes_trimmed = test_regimes[WINDOW_SIZE:]
    
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
    
    sim_date = times[0].strftime('%B %d, %Y')

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.15, 0.15])

    fig.add_trace(go.Scatter(x=times, y=actual_prices, name="Spot Price", 
                             line=dict(color='#00ff88', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=forecast_times, y=pred_prices_raw, name="Alpha Forecast (T+1)", 
                             line=dict(color='#ff00ff', dash='dot', width=1.5)), row=1, col=1)

    # Conviction Heatmap
    bar_colors = [f'rgba({"0, 255, 136" if s > 0 else "255, 75, 75"}, {c})' if s != 0 else 'rgba(100, 100, 100, 0.1)' 
                  for s, c in zip(active_signals, confidence_scores)]
    
    signal_labels = ["LONG" if s > 0 else "SHORT" if s < 0 else "NEUTRAL" for s in active_signals]
    
    fig.add_trace(go.Bar(x=times, y=[1]*len(active_signals), marker_color=bar_colors,
                         name="Signal State", showlegend=False,
                         text=signal_labels,
                         hovertemplate="<b>State:</b> %{text}<br><b>Conviction:</b> %{customdata:.1%}<extra></extra>",
                         customdata=confidence_scores), row=2, col=1)

    # --- REGIME HEATMAP ---
    # Legend mapping for sorted IDs
    regime_map = {0: "Quiet/Stable", 1: "Trending", 2: "Volatile/Chaos"}
    regime_colors = ['#444444', '#0099ff', '#ffcc00'] # Grey, Blue, Amber
    
    regime_names = [regime_map.get(int(r), "Unknown") for r in regimes_trimmed]

    fig.add_trace(go.Bar(x=times, y=[1]*len(regimes_trimmed), 
                         marker_color=[regime_colors[int(r)] for r in regimes_trimmed],
                         name="Market Regime", showlegend=False,
                         text=regime_names,
                         hovertemplate="<b>Regime:</b> %{text}<extra></extra>"), row=3, col=1)

    fig.update_layout(title=f"Alpha Analytics: {ticker} | Simulation Date: {sim_date}", 
                      template="plotly_dark", height=750, hovermode="x unified", margin=dict(t=50, b=50))
    
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Signal", showticklabels=False, row=2, col=1)
    fig.update_yaxes(title_text="Regime", showticklabels=False, row=3, col=1)
    
    st.plotly_chart(fig, width="stretch")

    # --- REGIME LEGEND ---
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"⬛ **Regime 0**: {regime_map[0]}")
    c2.markdown(f"🟦 **Regime 1**: {regime_map[1]}")
    c3.markdown(f"🟨 **Regime 2**: {regime_map[2]}")

    # EQUITY CURVE
    fig_equity = go.Figure()
    fig_equity.add_trace(go.Scatter(x=times, y=cum_returns * 100, fill='tozeroy', 
                                    name="Backtested Equity", line=dict(color='#3399ff', width=3)))
    fig_equity.update_layout(title="Net Performance Curve (%)", 
                             xaxis_title="Simulation Time (EST)", yaxis_title="Cumulative PnL (%)",
                             template="plotly_dark", height=400)
    st.plotly_chart(fig_equity, width="stretch")

except Exception as e:
    st.error(f"Backtest Runtime Error: {e}")


