# src/regime.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from .config import RANDOM_STATE

class RegimeDetector:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)

    def fit_predict(self, X: pd.DataFrame):
        features = X[["log_return", "volatility_20"]].fillna(0)
        return self.model.fit_predict(features)

    def predict(self, X: pd.DataFrame):
        features = X[["log_return", "volatility_20"]].fillna(0)
        return self.model.predict(features)

def add_regime_feature(X: pd.DataFrame, regimes):
    X = X.copy()
    X["regime"] = regimes
    return X