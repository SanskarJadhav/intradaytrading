# src/config.py

INTERVAL = "5m"
TRAIN_DAYS = 20  # Roughly one trading month
WINDOW_SIZE = 12 # 1 hour of lookback (12 * 5m) - academically superior for intraday
RANDOM_STATE = 42

# XGBoost: Adjusted for "Stochastic" intraday returns
XGB_PARAMS = {
    "n_estimators": 150,
    "max_depth": 3,      # Shallow trees are better for high-noise financial data
    "learning_rate": 0.05, 
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "random_state": RANDOM_STATE
}

# Regime Detection Settings
N_REGIMES = 3  # Bear, Bull, Sideways/Volatile
