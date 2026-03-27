"""
backend/models/inflation_model.py

Trains an XGBoost model to nowcast the US YoY inflation rate, using the same
feature set as the unemployment champion model (MacroMinds.ipynb Cell 15).

Public API
----------
train()
    Loads data, trains XGBoost targeting Inflation_Rate, prints RMSE + R²,
    saves the model to MODEL_PATH.  Returns the fitted model.

predict(features_dict)
    Loads the saved model and returns a single inflation rate prediction.

Usage
-----
    python -m backend.models.inflation_model
    python backend/models/inflation_model.py
"""

import os
import sys
import logging

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.data.preprocessing import build_features, MODEL_FEATURES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

TRAIN_CUTOFF = '2022-01-01'
TEST_END     = '2025-12-31'
TARGET       = 'Inflation_Rate'
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'inflation_xgb.pkl')


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

def _get_inflation_splits(
    cutoff_date: str,
    end_test_date: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build the full feature DataFrame and split on Inflation_Rate as the target.
    Uses build_features() directly because preprocessing.get_training_data()
    hardcodes Unemployment as the target.
    """
    df = build_features()

    # Ensure rows with a missing Inflation_Rate are excluded
    df = df.dropna(subset=MODEL_FEATURES + [TARGET])

    train = df.loc[:cutoff_date]
    test  = df.loc[cutoff_date:end_test_date]

    X_train = train[MODEL_FEATURES]
    y_train = train[TARGET]
    X_test  = test[MODEL_FEATURES]
    y_test  = test[TARGET]

    log.info(
        f"Train: {len(X_train)} rows  "
        f"[{X_train.index[0].date()} → {X_train.index[-1].date()}]"
    )
    log.info(
        f"Test : {len(X_test)} rows   "
        f"[{X_test.index[0].date()} → {X_test.index[-1].date()}]"
    )
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train() -> xgb.XGBRegressor:
    """
    Full training run:
      1. Load and split data (target = Inflation_Rate)
      2. Train XGBoost with the same hyperparameters as the notebook (Cell 15)
      3. Print RMSE and R²
      4. Save the model to MODEL_PATH
      5. Return the fitted model
    """
    log.info("=== Inflation Model Training ===")
    log.info(f"Split: train ≤ {TRAIN_CUTOFF}  |  test {TRAIN_CUTOFF} – {TEST_END}")

    X_train, y_train, X_test, y_test = _get_inflation_splits(TRAIN_CUTOFF, TEST_END)

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=3,
        objective='reg:squarederror',
        early_stopping_rounds=50,
        random_state=42,
    )

    log.info(f"Training XGBoost on {len(X_train)} rows...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)

    print("\n--- Inflation Model Results ---")
    print(f"{'Model':<22} {'RMSE':>8} {'R²':>8}")
    print("-" * 42)
    print(f"{'XGBoost':<22} {rmse:>8.4f} {r2:>8.4f}")
    print()

    joblib.dump(model, MODEL_PATH)
    log.info(f"Model saved → {MODEL_PATH}")

    return model


def predict(features_dict: dict) -> float:
    """
    Load the saved XGBoost model and return a single inflation rate prediction.

    Parameters
    ----------
    features_dict : dict
        Must contain the four model features (matching MODEL_FEATURES):
          - Claims_Z_Lag1
          - Income_Z_Lag1
          - Inflation_Lag1
          - Unemployment_Lag1

    Returns
    -------
    float — predicted YoY inflation rate (%)

    Raises
    ------
    FileNotFoundError if train() has not been run yet.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model found at {MODEL_PATH}. Run train() first."
        )

    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([features_dict])[MODEL_FEATURES]
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train()
