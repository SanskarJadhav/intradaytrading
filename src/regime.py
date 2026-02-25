# src/regime.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from .config import RANDOM_STATE

class RegimeDetector:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        self.feature_cols = []

    def _extract_features(self, X: pd.DataFrame):
        """
        Safely extracts relevant features for clustering. 
        Adjusted to handle varying column naming conventions.
        """
        # Search for any columns containing 'return' or 'vol'
        # This prevents the "None of [Index] are in [columns]" error
        cols = X.columns.tolist()
        target_cols = []
        
        # Priority 1: Exact matches
        for preferred in ["log_return", "volatility_20"]:
            if preferred in cols:
                target_cols.append(preferred)
        
        # Priority 2: Fuzzy matches if exact matches aren't found
        if len(target_cols) < 2:
            ret_col = [c for c in cols if "return" in c.lower()][:1]
            vol_col = [c for c in cols if "vol" in c.lower()][:1]
            target_cols = ret_col + vol_col

        if not target_cols:
            # Fallback: Just use the first two features available
            target_cols = cols[:2]
            
        return X[target_cols].fillna(0)

    def fit_predict(self, X: pd.DataFrame):
        features = self._extract_features(X)
        self.feature_cols = features.columns.tolist() # Remember what we used
        return self.model.fit_predict(features)

    def predict(self, X: pd.DataFrame):
        # Use the same columns used during fitting
        if not self.feature_cols:
            return self.fit_predict(X)
        
        features = X[self.feature_cols].fillna(0)
        return self.model.predict(features)

def add_regime_feature(X: pd.DataFrame, regimes):
    X = X.copy()
    X["regime"] = regimes
    return X
