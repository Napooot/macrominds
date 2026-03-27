"""
backend/routes/api.py

Flask Blueprint exposing the MacroMinds prediction API.

Routes
------
GET /api/predictions
    Returns the latest unemployment and inflation nowcasts using
    get_latest_features() and both model predict() functions.

GET /api/historical
    Returns historical economic data from the database.
    Optional query params: start_date, end_date  (YYYY-MM-DD)

GET /api/simulate
    What-if scenario prediction.
    Required query params: claims, inflation, income, prev_unemployment
    Returns unemployment and inflation predictions for the given inputs.
"""

import os
import sys
import logging

from flask import Blueprint, jsonify, request

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.data.preprocessing import get_latest_features, build_features, MODEL_FEATURES
from backend.models.unemployment_model import predict as predict_unemployment
from backend.models.inflation_model import predict as predict_inflation
from backend.db.db_utils import get_engine

log = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')


def _err(message: str, status: int = 400):
    return jsonify({"error": message}), status


# ---------------------------------------------------------------------------
# GET /api/predictions
# ---------------------------------------------------------------------------

@api_bp.route('/predictions', methods=['GET'])
def predictions():
    """
    Returns unemployment and inflation nowcasts for the most recent data point.

    Response
    --------
    {
        "date": "YYYY-MM-DD",
        "unemployment_prediction": float,
        "inflation_prediction": float,
        "features_used": { ... }
    }
    """
    try:
        features = get_latest_features()
    except Exception as exc:
        log.exception("Failed to load latest features")
        return _err(f"Feature loading failed: {exc}", 500)

    try:
        unemp_pred = predict_unemployment(features)
    except FileNotFoundError as exc:
        return _err(str(exc), 503)
    except Exception as exc:
        log.exception("Unemployment prediction failed")
        return _err(f"Unemployment prediction failed: {exc}", 500)

    try:
        inf_pred = predict_inflation(features)
    except FileNotFoundError as exc:
        return _err(str(exc), 503)
    except Exception as exc:
        log.exception("Inflation prediction failed")
        return _err(f"Inflation prediction failed: {exc}", 500)

    # Retrieve the date of the latest feature row from the DB
    try:
        df = build_features()
        latest_date = df.index[-1].date().isoformat()
    except Exception:
        latest_date = None

    return jsonify({
        "date": latest_date,
        "unemployment_prediction": round(unemp_pred, 4),
        "inflation_prediction": round(inf_pred, 4),
        "features_used": {k: round(v, 6) for k, v in features.items()},
    })


# ---------------------------------------------------------------------------
# GET /api/historical
# ---------------------------------------------------------------------------

@api_bp.route('/historical', methods=['GET'])
def historical():
    """
    Returns historical rows from the economic_data table.

    Query params (all optional)
    ---------------------------
    start_date : YYYY-MM-DD   default: no lower bound
    end_date   : YYYY-MM-DD   default: no upper bound

    Response
    --------
    {
        "count": int,
        "data": [
            {
                "date": "YYYY-MM-DD",
                "unemployment": float,
                "inflation_cpi": float,
                "inflation_rate": float,
                "weekly_claims": float,
                "personal_income": float,
                "income_growth": float,
                "gdp_growth": float
            },
            ...
        ]
    }
    """
    start_date = request.args.get('start_date')
    end_date   = request.args.get('end_date')

    # Build parameterised query
    conditions = []
    params: dict = {}

    if start_date:
        conditions.append("date >= :start_date")
        params['start_date'] = start_date
    if end_date:
        conditions.append("date <= :end_date")
        params['end_date'] = end_date

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    query = f"""
        SELECT date, unemployment, inflation_cpi, inflation_rate,
               weekly_claims, personal_income, income_growth, gdp_growth
        FROM economic_data
        {where}
        ORDER BY date ASC
    """

    try:
        from sqlalchemy import text
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text(query), params).mappings().all()
    except Exception as exc:
        log.exception("Database query failed")
        return _err(f"Database error: {exc}", 500)

    data = [
        {
            "date":            str(r["date"]),
            "unemployment":    r["unemployment"],
            "inflation_cpi":   r["inflation_cpi"],
            "inflation_rate":  r["inflation_rate"],
            "weekly_claims":   r["weekly_claims"],
            "personal_income": r["personal_income"],
            "income_growth":   r["income_growth"],
            "gdp_growth":      r["gdp_growth"],
        }
        for r in rows
    ]

    return jsonify({"count": len(data), "data": data})


# ---------------------------------------------------------------------------
# GET /api/simulate
# ---------------------------------------------------------------------------

@api_bp.route('/simulate', methods=['GET'])
def simulate():
    """
    What-if scenario: supply raw economic inputs and get model predictions.

    The route replicates the notebook's run_simulation() (Cell 12) by
    z-scoring the raw claims and income values using the full-history
    mean/std from the feature dataset, then calling both models.

    Required query params
    ---------------------
    claims             : float  — raw weekly initial claims (e.g. 250000)
    inflation          : float  — current YoY inflation rate % (e.g. 3.5)
    income             : float  — current YoY income growth % (e.g. 2.1)
    prev_unemployment  : float  — previous month unemployment rate % (e.g. 4.0)

    Response
    --------
    {
        "inputs": { ... },
        "features": { ... },
        "unemployment_prediction": float,
        "inflation_prediction": float
    }
    """
    required = ['claims', 'inflation', 'income', 'prev_unemployment']
    missing  = [p for p in required if request.args.get(p) is None]
    if missing:
        return _err(f"Missing required query params: {', '.join(missing)}")

    try:
        claims_raw    = float(request.args['claims'])
        inflation_raw = float(request.args['inflation'])
        income_raw    = float(request.args['income'])
        prev_unemp    = float(request.args['prev_unemployment'])
    except ValueError:
        return _err("All query params must be numeric")

    # Z-score the raw claims and income using full-history stats
    try:
        df = build_features()
    except Exception as exc:
        log.exception("Failed to build feature dataset for simulation")
        return _err(f"Feature dataset error: {exc}", 500)

    claims_mean = df['Weekly_Claims'].mean()
    claims_std  = df['Weekly_Claims'].std()
    income_mean = df['Income_Growth'].mean()
    income_std  = df['Income_Growth'].std()

    features = {
        'Claims_Z_Lag1':       (claims_raw - claims_mean) / claims_std,
        'Income_Z_Lag1':       (income_raw - income_mean) / income_std,
        'Inflation_Lag1':      inflation_raw,
        'Unemployment_Lag1':   prev_unemp,
    }

    try:
        unemp_pred = predict_unemployment(features)
    except FileNotFoundError as exc:
        return _err(str(exc), 503)
    except Exception as exc:
        log.exception("Simulation unemployment prediction failed")
        return _err(f"Unemployment prediction failed: {exc}", 500)

    try:
        inf_pred = predict_inflation(features)
    except FileNotFoundError as exc:
        return _err(str(exc), 503)
    except Exception as exc:
        log.exception("Simulation inflation prediction failed")
        return _err(f"Inflation prediction failed: {exc}", 500)

    return jsonify({
        "inputs": {
            "claims":            claims_raw,
            "inflation":         inflation_raw,
            "income":            income_raw,
            "prev_unemployment": prev_unemp,
        },
        "features": {k: round(v, 6) for k, v in features.items()},
        "unemployment_prediction": round(unemp_pred, 4),
        "inflation_prediction":    round(inf_pred, 4),
    })
