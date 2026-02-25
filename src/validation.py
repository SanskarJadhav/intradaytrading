# src/validation.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .models import scale_features
from .ensemble import StackedEnsemble
from .regime import RegimeDetector, add_regime_feature

def walk_forward_validation(X_df, y_series, n_splits=3):
    """
    Simulating real-time deployment by testing on chronological chunks.
    'Out-of-Sample' robustness and 'Regime Adaptation'.
    """
    total_len = len(X_df)
    fold_size = total_len // (n_splits + 1)
    metrics = []

    for i in range(1, n_splits + 1):
        # academic standard is training window expands, test window follows
        train_end = fold_size * i
        test_end = train_end + fold_size

        X_train_fold = X_df.iloc[:train_end]
        y_train_fold = y_series.iloc[:train_end]
        X_test_fold = X_df.iloc[train_end:test_end]
        y_test_fold = y_series.iloc[train_end:test_end]

        # 1. Regime Detection
        rd = RegimeDetector()
        train_regimes = rd.fit_predict(X_train_fold)
        test_regimes = rd.predict(X_test_fold)
        
        X_train_reg = add_regime_feature(X_train_fold, train_regimes)
        X_test_reg = add_regime_feature(X_test_fold, test_regimes)

        # 2. Sequential Scaling
        X_tr_s, X_te_s, _ = scale_features(X_train_reg.values, X_test_reg.values)

        # 3. Model Fit & Prediction
        model = StackedEnsemble()
        model.fit(X_tr_s, y_train_fold.values)
        preds = model.predict(X_te_s)

        # 4. Error Attribution
        # MAE and RMSE are standard, but for alpha, direction matters most
        y_true = y_test_fold.values
        metrics.append({
            "fold": i,
            "period": f"{X_test_fold.index[0].date()} to {X_test_fold.index[-1].date()}",
            "mae": mean_absolute_error(y_true, preds),
            "rmse": np.sqrt(mean_squared_error(y_true, preds))
        })

    return metrics

