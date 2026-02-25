# src/config.py

INTERVAL = "5m"
TRAIN_DAYS = 20
WINDOW_SIZE = 10 # Increased lookback for better feature capture
RANDOM_STATE = 42

LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 32

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.01, # Slower learning to prevent overfitting returns
    "subsample": 0.8,
    "random_state": RANDOM_STATE
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}