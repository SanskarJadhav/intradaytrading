# src/features.py

import pandas as pd
import numpy as np

def build_feature_matrix(df: pd.DataFrame):
    df = df.copy()
    
    # Simple percentage return
    df["ret"] = df["Close"].pct_change()
    df["vol_20"] = df["ret"].rolling(20).std()
    
    # RSI Calculation
    delta = df["Close"].diff()
    up = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    down = -1 * delta.clip(upper=0).ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + (up / down.replace(0, np.nan))))
    
    # Target: The percentage return of the NEXT 5-minute bar
    df["target"] = df["ret"].shift(-1)
    
    df = df.dropna().replace([np.inf, -np.inf], 0)
    
    feature_cols = ["ret", "vol_20", "rsi", "Volume"]
    return df[feature_cols], df["target"]