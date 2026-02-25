# src/risk.py

import numpy as np

def compute_strategy_returns(y_true, y_pred):
    """
    Directional strategy: Long if predicted price change > 0, else short.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[-min_len:]
    y_pred = y_pred[-min_len:]

    # Calculate price changes
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    # Position is 1 (long), -1 (short), or 0 (neutral if no change predicted)
    positions = np.sign(pred_diff)
    
    # Strategy return = position * actual percentage change 
    # (Using simple diff here to match your logic, but scaled to return)
    strategy_returns = positions * true_diff
    return strategy_returns

def sharpe_ratio(returns):
    returns = np.asarray(returns).flatten()
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    # Annualization factor for 5-minute data (approximate)
    # 78 intervals per day * 252 days
    return (np.mean(returns) / np.std(returns)) * np.sqrt(78 * 252)

def max_drawdown(returns):
    returns = np.asarray(returns).flatten()
    if len(returns) == 0: return 0.0
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return np.max(drawdown)

def hit_ratio(y_true, y_pred):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[-min_len:]
    y_pred = y_pred[-min_len:]
    
    t_diff = np.sign(np.diff(y_true))
    p_diff = np.sign(np.diff(y_pred))
    return np.mean(t_diff == p_diff)