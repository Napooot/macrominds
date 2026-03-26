-- MacroMinds Database Schema

-- Core table: stores all economic time series data
CREATE TABLE IF NOT EXISTS economic_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    unemployment FLOAT,
    inflation_cpi FLOAT,
    inflation_rate FLOAT,
    weekly_claims FLOAT,
    personal_income FLOAT,
    income_growth FLOAT,
    gdp_growth FLOAT,
    source VARCHAR(20) DEFAULT 'FRED',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(date, source)
);

-- Stores model predictions for tracking accuracy
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    target_variable VARCHAR(50) NOT NULL,
    predicted_value FLOAT,
    actual_value FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tracks trained models and their performance
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version INT DEFAULT 1,
    rmse FLOAT,
    r2_score FLOAT,
    features TEXT,
    hyperparameters JSONB,
    file_path VARCHAR(255),
    trained_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast date lookups
CREATE INDEX IF NOT EXISTS idx_economic_data_date ON economic_data(date);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date);
