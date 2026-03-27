"""
backend/models/unemployment_model.py

Trains an XGBoost (champion) and ARIMA (comparison baseline) model to nowcast
the US unemployment rate, mirroring the champion model in MacroMinds.ipynb
(Cells 9–10).

Public API
----------
train()
    Loads data, trains both models, prints RMSE + R², saves the best model.
    Returns the fitted XGBoost model (always saved as the prediction model).

predict(features_dict)
    Loads the saved XGBoost model and returns a single unemployment prediction.

Usage
-----
    python -m backend.models.unemployment_model
    python backend/models/unemployment_model.py
"""

import os
import sys
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.data.preprocessing import get_training_data, MODEL_FEATURES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
warnings.filterwarnings('ignore')   # suppress ARIMA convergence noise

TRAIN_CUTOFF = '2022-01-01'
TEST_END     = '2025-12-31'
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'unemployment_xgb.pkl')


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> tuple[xgb.XGBRegressor, np.ndarray, float, float]:
    """
    XGBoost with the same hyperparameters as the champion model in the notebook
    (Cell 9): n_estimators=1000, lr=0.05, max_depth=3, early stopping on test.
    """
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=3,
        objective='reg:squarederror',
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)
    return model, preds, rmse, r2


def _train_arima(
    y_train: pd.Series,
    y_test:  pd.Series,
) -> tuple[object, np.ndarray, float, float]:
    """
    ARIMA(2,1,2) univariate baseline fitted on y_train; forecasts len(y_test) steps.
    Returns (fitted_result, predictions, rmse, r2).
    On convergence failure the function returns NaN metrics so training can
    continue without crashing.
    """
    # ARIMA needs a DatetimeIndex with uniform frequency
    y_tr = y_train.copy()
    y_tr.index = pd.DatetimeIndex(y_tr.index).to_period('M').to_timestamp()

    try:
        result = ARIMA(y_tr, order=(2, 1, 2)).fit()
        preds  = np.array(result.forecast(steps=len(y_test)))
        rmse   = np.sqrt(mean_squared_error(y_test.values, preds))
        r2     = r2_score(y_test.values, preds)
    except Exception as exc:
        log.warning(f"ARIMA fitting failed: {exc}")
        preds  = np.full(len(y_test), np.nan)
        rmse   = np.nan
        r2     = np.nan
        result = None

    return result, preds, rmse, r2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train() -> xgb.XGBRegressor:
    """
    Full training run:
      1. Load train/test data from preprocessing.get_training_data()
      2. Train XGBoost (champion) and ARIMA(2,1,2) (comparison baseline)
      3. Print RMSE and R² for both models
      4. Save the XGBoost model to MODEL_PATH (always used for live prediction)
      5. Return the fitted XGBoost model

    The XGBoost model is always saved regardless of which model "wins" on RMSE,
    because predict() requires feature-based inference that ARIMA cannot support.
    """
    log.info("=== Unemployment Model Training ===")
    log.info(f"Split: train ≤ {TRAIN_CUTOFF}  |  test {TRAIN_CUTOFF} – {TEST_END}")

    X_train, y_train, X_test, y_test = get_training_data(TRAIN_CUTOFF, TEST_END)

    log.info(f"Training XGBoost on {len(X_train)} rows...")
    xgb_model, _, xgb_rmse, xgb_r2 = _train_xgboost(X_train, y_train, X_test, y_test)

    log.info(f"Training ARIMA(2,1,2) on {len(y_train)} rows...")
    _, _, arima_rmse, arima_r2 = _train_arima(y_train, y_test)

    # --- Results table ---
    print("\n--- Unemployment Model Results ---")
    print(f"{'Model':<22} {'RMSE':>8} {'R²':>8}")
    print("-" * 42)
    print(f"{'XGBoost (champion)':<22} {xgb_rmse:>8.4f} {xgb_r2:>8.4f}")
    if not np.isnan(arima_rmse):
        print(f"{'ARIMA(2,1,2)':<22} {arima_rmse:>8.4f} {arima_r2:>8.4f}")
    else:
        print(f"{'ARIMA(2,1,2)':<22} {'failed':>8} {'—':>8}")

    winner = (
        "XGBoost" if np.isnan(arima_rmse) or xgb_rmse <= arima_rmse else "ARIMA"
    )
    print(f"\nBest model: {winner}")
    print()

    # --- Save XGBoost (feature-based, required for predict()) ---
    joblib.dump(xgb_model, MODEL_PATH)
    log.info(f"XGBoost saved → {MODEL_PATH}")

    return xgb_model


def predict(features_dict: dict) -> float:
    """
    Load the saved XGBoost model and return a single unemployment prediction.

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
    float — predicted unemployment rate (%)

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
