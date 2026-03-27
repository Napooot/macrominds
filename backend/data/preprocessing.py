"""
backend/data/preprocessing.py

Reads from the economic_data table and produces model-ready feature matrices,
replicating the feature engineering pipeline from MacroMinds.ipynb (Cells 3, 7, 9).

Public API
----------
build_features()
    Reads DB, engineers all features, drops NaN rows, prints summary.
    Returns a fully-featured DataFrame indexed by date.

get_training_data(cutoff_date, end_test_date)
    Returns (X_train, y_train, X_test, y_test) for a time-series split.
    • train : start → cutoff_date
    • test  : cutoff_date → end_test_date

get_latest_features()
    Returns the most recent row's feature values as a dict, ready for
    passing straight into a model.predict() call.

Usage
-----
    python -m backend.data.preprocessing          # from project root
    python backend/data/preprocessing.py          # direct
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

# Allow running as a standalone script from any working directory
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.db.db_utils import get_engine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# Features the notebook's champion model (XGBoost + autoregression) expects
MODEL_FEATURES = [
    'Claims_Z_Lag1',
    'Income_Z_Lag1',
    'Inflation_Lag1',
    'Unemployment_Lag1',
]
TARGET = 'Unemployment'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_from_db() -> pd.DataFrame:
    """
    Read all rows from economic_data, rename columns to the canonical names
    used throughout the notebook, and sort by date.

    DB column       → canonical name
    ----------------------------------------
    unemployment    → Unemployment
    inflation_cpi   → Inflation
    inflation_rate  → Inflation_Rate   (YoY, already stored by ingestion)
    weekly_claims   → Weekly_Claims
    personal_income → Personal_Income
    income_growth   → Income_Growth    (YoY, already stored by ingestion)
    gdp_growth      → GDP_Growth
    """
    engine = get_engine()
    query = """
        SELECT date, unemployment, inflation_cpi, inflation_rate,
               weekly_claims, personal_income, income_growth, gdp_growth
        FROM economic_data
        ORDER BY date ASC
    """
    df = pd.read_sql(query, engine, parse_dates=['date'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)

    df.rename(columns={
        'unemployment':    'Unemployment',
        'inflation_cpi':   'Inflation',
        'inflation_rate':  'Inflation_Rate',
        'weekly_claims':   'Weekly_Claims',
        'personal_income': 'Personal_Income',
        'income_growth':   'Income_Growth',
        'gdp_growth':      'GDP_Growth',
    }, inplace=True)

    log.info(
        f"Loaded {len(df)} rows from economic_data  "
        f"[{df.index[0].date()} → {df.index[-1].date()}]"
    )
    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline from MacroMinds.ipynb.

    Steps
    -----
    1. YoY rates — use stored Inflation_Rate / Income_Growth where present;
       recompute from raw columns for any rows where they are NaN (handles
       edge cases if ingestion left gaps).
    2. Lag features (t-1) : Claims_Lag1, Inflation_Lag1, Income_Lag1,
                             Unemployment_Lag1
    3. Z-score normalisation (full-history mean/std):
                             Unemployment_Z, Weekly_Claims_Z, Income_Growth_Z
    4. Z-score lag features : Claims_Z_Lag1, Income_Z_Lag1
    5. Drop any rows that still contain NaN in the required feature columns.
    """
    df = df.copy()

    # --- 1. YoY rates (fill gaps if DB rows are missing them) ---
    if df['Inflation_Rate'].isna().any():
        computed = df['Inflation'].pct_change(12) * 100
        df['Inflation_Rate'] = df['Inflation_Rate'].fillna(computed)

    if df['Income_Growth'].isna().any():
        computed = df['Personal_Income'].pct_change(12) * 100
        df['Income_Growth'] = df['Income_Growth'].fillna(computed)

    # --- 2. Lag features (t-1) ---
    df['Claims_Lag1']       = df['Weekly_Claims'].shift(1)
    df['Inflation_Lag1']    = df['Inflation_Rate'].shift(1)
    df['Income_Lag1']       = df['Income_Growth'].shift(1)
    df['Unemployment_Lag1'] = df['Unemployment'].shift(1)

    # --- 3. Z-score normalisation (full-history stats) ---
    for col in ['Unemployment', 'Weekly_Claims', 'Income_Growth']:
        mean, std = df[col].mean(), df[col].std()
        df[f'{col}_Z'] = (df[col] - mean) / std

    # --- 4. Z-score lag features ---
    df['Claims_Z_Lag1'] = df['Weekly_Claims_Z'].shift(1)
    df['Income_Z_Lag1'] = df['Income_Growth_Z'].shift(1)

    # --- 5. Drop rows with NaN in any model-required column ---
    required = MODEL_FEATURES + [TARGET]
    before = len(df)
    df.dropna(subset=required, inplace=True)
    dropped = before - len(df)
    if dropped:
        log.info(f"Dropped {dropped} rows with NaN in required columns")

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features() -> pd.DataFrame:
    """
    Load the DB, engineer all features, drop NaN rows, and print a summary.
    Returns the fully-featured DataFrame indexed by date.
    """
    log.info("=== MacroMinds Preprocessing ===")

    df_raw = _load_from_db()
    df = _engineer(df_raw)

    print("\n--- Dataset Summary ---")
    print(f"Shape      : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Columns    : {list(df.columns)}")
    print(f"\nNull counts (should all be 0 for model features):")
    null_counts = df[MODEL_FEATURES + [TARGET]].isna().sum()
    print(null_counts.to_string())
    print()

    return df


def get_training_data(
    cutoff_date: str,
    end_test_date: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build features then split into train / test sets using a time-series cut.

    Parameters
    ----------
    cutoff_date   : str  — last date (inclusive) of the training set, e.g. '2015-01-01'
    end_test_date : str  — last date (inclusive) of the test set,     e.g. '2019-12-31'

    Returns
    -------
    X_train, y_train, X_test, y_test
        X_* : DataFrame of MODEL_FEATURES columns
        y_* : Series of TARGET (Unemployment)
    """
    df = build_features()

    train = df.loc[:cutoff_date]
    test  = df.loc[cutoff_date:end_test_date]

    X_train = train[MODEL_FEATURES]
    y_train = train[TARGET]
    X_test  = test[MODEL_FEATURES]
    y_test  = test[TARGET]

    log.info(
        f"Train: {len(X_train)} rows  [{X_train.index[0].date()} → {X_train.index[-1].date()}]"
    )
    log.info(
        f"Test : {len(X_test)} rows   [{X_test.index[0].date()} → {X_test.index[-1].date()}]"
    )

    return X_train, y_train, X_test, y_test


def get_latest_features() -> dict:
    """
    Return the most recent row's MODEL_FEATURES values as a dict.
    Suitable for direct use in model.predict(pd.DataFrame([get_latest_features()])).

    Returns
    -------
    dict with keys: Claims_Z_Lag1, Income_Z_Lag1, Inflation_Lag1, Unemployment_Lag1
    """
    df = build_features()
    latest = df[MODEL_FEATURES].iloc[-1]

    log.info(f"Latest feature row: {latest.name.date()}")
    log.info(f"  {latest.to_dict()}")

    return latest.to_dict()


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("--- build_features() ---")
    df_full = build_features()

    print("--- get_training_data('2015-01-01', '2019-12-31') ---")
    X_tr, y_tr, X_te, y_te = get_training_data('2015-01-01', '2019-12-31')
    print(f"X_train: {X_tr.shape}  y_train: {y_tr.shape}")
    print(f"X_test : {X_te.shape}  y_test : {y_te.shape}")
    print(X_tr.tail(3))

    print("\n--- get_latest_features() ---")
    feats = get_latest_features()
    print(feats)
