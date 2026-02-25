# src/models.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import StandardScaler

WINDOW_SIZE = 6 # 30 minutes lookback

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler

def create_sequences(X, y):
    xs, ys = [], []
    for i in range(WINDOW_SIZE, len(X)):
        xs.append(X[i - WINDOW_SIZE:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)

def train_lstm(X_seq, y_seq):
    model = Sequential([
        Input(shape=(WINDOW_SIZE, X_seq.shape[2])),
        LSTM(16, return_sequences=False),
        Dense(1, activation='linear') 
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    model.fit(X_seq, y_seq, epochs=10, batch_size=32, verbose=0)
    return model