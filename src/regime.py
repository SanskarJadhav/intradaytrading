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
        self.rank_map = {} # Maps random k-means IDs to sorted Volatility IDs

    def _extract_features(self, X: pd.DataFrame):
        cols = X.columns.tolist()
        target_cols = []
        
        # Priority 1: Exact matches
        for preferred in ["log_return", "volatility_20"]:
            if preferred in cols:
                target_cols.append(preferred)
        
        # Priority 2: Fuzzy matches
        if len(target_cols) < 2:
            ret_col = [c for c in cols if "return" in c.lower()][:1]
            vol_col = [c for c in cols if "vol" in c.lower()][:1]
            target_cols = ret_col + vol_col

        if not target_cols:
            target_cols = cols[:2]
            
        return X[target_cols].fillna(0)

    def fit_predict(self, X: pd.DataFrame):
        features = self._extract_features(X)
        self.feature_cols = features.columns.tolist()
        
        # Initial fit
        raw_labels = self.model.fit_predict(features)
        
        # --- SORTING LOGIC ---
        # Find which column represents volatility
        vol_idx = 0
        for i, col in enumerate(self.feature_cols):
            if "vol" in col.lower():
                vol_idx = i
                break
        
        # Get the centers of the clusters for the volatility column
        centers = self.model.cluster_centers_[:, vol_idx]
        
        # Create a mapping: Sort centers ascending (Low Vol -> High Vol)
        # np.argsort(centers) gives the indices of the centers from lowest to highest
        sorted_indices = np.argsort(centers)
        self.rank_map = {raw_id: rank for rank, raw_id in enumerate(sorted_indices)}
        
        # Return sorted labels
        return np.array([self.rank_map[label] for label in raw_labels])

    def predict(self, X: pd.DataFrame):
        if not self.feature_cols:
            return self.fit_predict(X)
        
        features = X[self.feature_cols].fillna(0)
        raw_labels = self.model.predict(features)
        
        # Apply the learned rank mapping
        return np.array([self.rank_map[label] for label in raw_labels])

def add_regime_feature(X: pd.DataFrame, regimes):
    X = X.copy()
    X["regime"] = regimes
    return X
