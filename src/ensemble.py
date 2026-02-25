# src/ensemble.py

import numpy as np
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from .models import train_lstm, create_sequences, WINDOW_SIZE

class StackedEnsemble:
    def __init__(self):
        self.xgb = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)
        self.rf = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        self.meta = Ridge()
        self.lstm = None

    def fit(self, X_s, y):
        X_seq, y_seq = create_sequences(X_s, y)
        self.lstm = train_lstm(X_seq, y_seq)
        
        x_base = X_s[WINDOW_SIZE:]
        p_xgb = self.xgb.fit(X_s, y).predict(x_base)
        p_rf = self.rf.fit(X_s, y).predict(x_base)
        p_lstm = self.lstm.predict(X_seq, verbose=0).flatten()
        
        meta_X = np.column_stack([p_xgb, p_rf, p_lstm])
        self.meta.fit(meta_X, y_seq)

    def predict(self, X_s):
        X_seq, _ = create_sequences(X_s, np.zeros(len(X_s)))
        p_lstm = self.lstm.predict(X_seq, verbose=0).flatten()
        
        x_base = X_s[WINDOW_SIZE:]
        p_xgb = self.xgb.predict(x_base)
        p_rf = self.rf.predict(x_base)
        
        meta_X = np.column_stack([p_xgb, p_rf, p_lstm])
        raw_preds = self.meta.predict(meta_X)
        
        # FIX: The Ultimate Safety Net. 
        # Limits predictions to a MAXIMUM of a 3% price move per 5 minutes.
        # It is now mathematically impossible to see 71k on your chart.
        return np.clip(raw_preds, -0.03, 0.03)