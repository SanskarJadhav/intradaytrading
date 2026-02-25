# src/validation.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .models import scale_features
from .ensemble import StackedEnsemble
from .regime import RegimeDetector, add_regime_feature

def walk_forward_validation(X_df, y_series, n_splits=3):
    """
    Walk-forward validation using DataFrames to preserve feature names for Regime Detection.
    """
    fold_size = len(X_df) // (n_splits + 1)
    metrics = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)

        X_train_fold = X_df.iloc[:train_end]
        y_train_fold = y_series.iloc[:train_end]
        X_test_fold = X_df.iloc[train_end:test_end]
        y_test_fold = y_series.iloc[train_end:test_end]

        # 1. Regime Detection for this fold
        rd = RegimeDetector()
        train_regimes = rd.fit_predict(X_train_fold)
        test_regimes = rd.predict(X_test_fold)
        
        X_train_reg = add_regime_feature(X_train_fold, train_regimes)
        X_test_reg = add_regime_feature(X_test_fold, test_regimes)

        # 2. Scaling
        X_tr_s, X_te_s, _ = scale_features(X_train_reg.values, X_test_reg.values)

        # 3. Model
        model = StackedEnsemble()
        model.fit(X_tr_s, y_train_fold.values)
        preds = model.predict(X_te_s)

        # 4. Metrics
        y_test_trim = y_test_fold.values[-len(preds):]
        mae = mean_absolute_error(y_test_trim, preds)
        
        metrics.append({
            "fold": i,
            "mae": mae,
            "rmse": np.sqrt(mean_squared_error(y_test_trim, preds))
        })

    return metrics