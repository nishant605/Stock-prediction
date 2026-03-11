"""
prediction_engine.py — Stock Prediction Backend
=================================================
Clean prediction logic used by app_pro.py
Does NOT modify any existing files.
"""

import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "Copy of df_clean.pkl")
LOOKBACK = 60

FEATURES = [
    'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume',
    'Deliverable Volume',
    'return', 'MA10', 'MA50', 'volatility',
    'MACD', 'High_Low_Ratio', 'Close_Open_Ratio'
]

STOCK_NAMES = {
    'ADANIPORTS': 'Adani Ports & SEZ', 'ASIANPAINT': 'Asian Paints',
    'AXISBANK': 'Axis Bank', 'BAJAJ-AUTO': 'Bajaj Auto',
    'BAJAJFINSV': 'Bajaj Finserv', 'BAJAUTOFIN': 'Bajaj Auto Finance',
    'BAJFINANCE': 'Bajaj Finance', 'BHARTIARTL': 'Bharti Airtel',
    'BPCL': 'Bharat Petroleum', 'BRITANNIA': 'Britannia Industries',
    'CIPLA': 'Cipla', 'COALINDIA': 'Coal India',
    'DRREDDY': "Dr. Reddy's Labs", 'EICHERMOT': 'Eicher Motors',
    'GAIL': 'GAIL India', 'GRASIM': 'Grasim Industries',
    'HCLTECH': 'HCL Technologies', 'HDFC': 'HDFC Ltd',
    'HDFCBANK': 'HDFC Bank', 'HEROMOTOCO': 'Hero MotoCorp',
    'HINDALCO': 'Hindalco Industries', 'HINDUNILVR': 'Hindustan Unilever',
    'ICICIBANK': 'ICICI Bank', 'INDUSINDBK': 'IndusInd Bank',
    'INFY': 'Infosys', 'IOC': 'Indian Oil Corp',
    'ITC': 'ITC Ltd', 'JSWSTEEL': 'JSW Steel',
    'KOTAKBANK': 'Kotak Mahindra Bank', 'LT': 'Larsen & Toubro',
    'M&M': 'Mahindra & Mahindra', 'MARUTI': 'Maruti Suzuki',
    'NESTLEIND': 'Nestle India', 'NTPC': 'NTPC Ltd',
    'ONGC': 'ONGC', 'POWERGRID': 'Power Grid Corp',
    'RELIANCE': 'Reliance Industries', 'SBIN': 'State Bank of India',
    'SHREECEM': 'Shree Cement', 'SUNPHARMA': 'Sun Pharma',
    'TATAMOTORS': 'Tata Motors', 'TATASTEEL': 'Tata Steel',
    'TCS': 'TCS', 'TECHM': 'Tech Mahindra',
    'TITAN': 'Titan Company', 'ULTRACEMCO': 'UltraTech Cement',
    'UPL': 'UPL Ltd', 'VEDL': 'Vedanta Ltd',
    'WIPRO': 'Wipro', 'ZEEL': 'Zee Entertainment',
}

STOCK_SECTORS = {
    'ADANIPORTS': 'Infrastructure', 'ASIANPAINT': 'Consumer', 'AXISBANK': 'Banking',
    'BAJAJ-AUTO': 'Automobile', 'BAJAJFINSV': 'Finance', 'BAJAUTOFIN': 'Finance',
    'BAJFINANCE': 'Finance', 'BHARTIARTL': 'Telecom', 'BPCL': 'Oil & Gas',
    'BRITANNIA': 'FMCG', 'CIPLA': 'Pharma', 'COALINDIA': 'Mining',
    'DRREDDY': 'Pharma', 'EICHERMOT': 'Automobile', 'GAIL': 'Oil & Gas',
    'GRASIM': 'Cement', 'HCLTECH': 'IT', 'HDFC': 'Finance',
    'HDFCBANK': 'Banking', 'HEROMOTOCO': 'Automobile', 'HINDALCO': 'Metals',
    'HINDUNILVR': 'FMCG', 'ICICIBANK': 'Banking', 'INDUSINDBK': 'Banking',
    'INFY': 'IT', 'IOC': 'Oil & Gas', 'ITC': 'FMCG',
    'JSWSTEEL': 'Metals', 'KOTAKBANK': 'Banking', 'LT': 'Infrastructure',
    'M&M': 'Automobile', 'MARUTI': 'Automobile', 'NESTLEIND': 'FMCG',
    'NTPC': 'Power', 'ONGC': 'Oil & Gas', 'POWERGRID': 'Power',
    'RELIANCE': 'Conglomerate', 'SBIN': 'Banking', 'SHREECEM': 'Cement',
    'SUNPHARMA': 'Pharma', 'TATAMOTORS': 'Automobile', 'TATASTEEL': 'Metals',
    'TCS': 'IT', 'TECHM': 'IT', 'TITAN': 'Consumer',
    'ULTRACEMCO': 'Cement', 'UPL': 'Chemicals', 'VEDL': 'Metals',
    'WIPRO': 'IT', 'ZEEL': 'Media',
}


def load_data():
    return pd.read_pickle(DATA_PATH)


def get_available_stocks():
    stocks = []
    for f in os.listdir(MODELS_DIR):
        if f.startswith("gru_model_") and f.endswith(".keras"):
            symbol = f.replace("gru_model_", "").replace(".keras", "")
            stocks.append(symbol)
    return sorted(stocks)


def get_display_name(symbol):
    return STOCK_NAMES.get(symbol, symbol)


def get_sector(symbol):
    return STOCK_SECTORS.get(symbol, 'Other')


def create_features(df, symbol):
    data = df[df['Symbol'] == symbol].copy()
    data = data.sort_values('Date').reset_index(drop=True)

    data['return'] = data['Close'].pct_change()
    data['MA10'] = data['Close'].rolling(10).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['volatility'] = data['Close'].rolling(10).std()
    data = data.dropna().reset_index(drop=True)

    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = ema12 - ema26
    data['High_Low_Ratio'] = data['High'] / (data['Low'] + 1e-10)
    data['Close_Open_Ratio'] = data['Close'] / (data['Open'] + 1e-10)
    data['target'] = data['Close'].pct_change().shift(-1)
    data['current_close'] = data['Close']
    data = data.dropna().reset_index(drop=True)
    return data


def load_stock_model(symbol):
    model = load_model(os.path.join(MODELS_DIR, f"gru_model_{symbol}.keras"))
    sc_X = pickle.load(open(os.path.join(MODELS_DIR, f"scaler_X_{symbol}.pkl"), "rb"))
    sc_y = pickle.load(open(os.path.join(MODELS_DIR, f"scaler_y_{symbol}.pkl"), "rb"))
    return model, sc_X, sc_y


def predict_next_day(data, model, scaler_X, scaler_y):
    X_scaled = scaler_X.transform(data[FEATURES])
    X_input = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    pred_scaled = model.predict(X_input[-1:], verbose=0)
    pred_return = scaler_y.inverse_transform(pred_scaled)[0][0]

    current_price = data['current_close'].iloc[-1]
    predicted_price = current_price * (1 + pred_return)

    return {
        'predicted_return': pred_return,
        'predicted_price': predicted_price,
        'current_price': current_price,
        'direction': 'UP' if pred_return > 0 else 'DOWN',
        'change_pct': pred_return * 100,
        'change_amount': predicted_price - current_price,
    }


def predict_multiple_days(data, model, scaler_X, scaler_y, num_days=5):
    recent_data = data.copy()
    predictions = []
    last_price = data['current_close'].iloc[-1]
    last_date = data['Date'].iloc[-1]

    X_scaled = scaler_X.transform(recent_data[FEATURES])
    X_input = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    for day in range(num_days):
        pred_scaled = model.predict(X_input[-1:], verbose=0)
        pred_return = scaler_y.inverse_transform(pred_scaled)[0][0]
        next_price = last_price * (1 + pred_return)

        predictions.append({
            'day': day + 1,
            'predicted_price': next_price,
            'predicted_return': pred_return,
            'change_pct': pred_return * 100,
        })

        # Update data for next iteration
        new_row = recent_data.iloc[-1].copy()
        new_row['Open'] = last_price
        new_row['Close'] = next_price
        new_row['High'] = next_price * 1.005
        new_row['Low'] = next_price * 0.995
        new_row['VWAP'] = next_price
        new_row['return'] = pred_return

        recent_data = pd.concat([recent_data, new_row.to_frame().T], ignore_index=True)
        recent_data['MA10'] = recent_data['Close'].rolling(10).mean()
        recent_data['MA50'] = recent_data['Close'].rolling(50).mean()
        recent_data['volatility'] = recent_data['Close'].rolling(10).std()
        ema12 = recent_data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = recent_data['Close'].ewm(span=26, adjust=False).mean()
        recent_data['MACD'] = ema12 - ema26
        recent_data['High_Low_Ratio'] = recent_data['High'] / (recent_data['Low'] + 1e-10)
        recent_data['Close_Open_Ratio'] = recent_data['Close'] / (recent_data['Open'] + 1e-10)
        recent_data = recent_data.ffill()

        X_scaled = scaler_X.transform(recent_data[FEATURES])
        X_input = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        last_price = next_price

    future_dates = pd.bdate_range(start=last_date, periods=num_days + 1)[1:]
    for i, pred in enumerate(predictions):
        pred['date'] = future_dates[i] if i < len(future_dates) else last_date + pd.Timedelta(days=i + 1)

    return predictions


def get_backtest_results(data, model, scaler_X, scaler_y, test_ratio=0.2):
    X_scaled = scaler_X.transform(data[FEATURES])
    X_scaled_seq = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    train_size = int(len(data) * (1 - test_ratio))
    X_test = X_scaled_seq[train_size:]
    y_test = data['target'].values[train_size:]
    dates_test = data['Date'].values[train_size:]
    closes_test = data['current_close'].values[train_size:]

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_returns = scaler_y.inverse_transform(y_pred_scaled).flatten()

    actual_prices = closes_test * (1 + y_test)
    predicted_prices = closes_test * (1 + y_pred_returns)

    mae = np.mean(np.abs(predicted_prices - actual_prices))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / (actual_prices + 1e-10))) * 100

    if len(actual_prices) > 1:
        actual_dir = np.sign(np.diff(actual_prices))
        pred_dir = np.sign(np.diff(predicted_prices))
        dir_accuracy = np.mean(actual_dir == pred_dir) * 100
    else:
        dir_accuracy = 0.0

    comparison_df = pd.DataFrame({
        'Date': dates_test,
        'Actual Price': actual_prices,
        'Predicted Price': predicted_prices,
    })

    return {
        'mae': mae, 'mape': mape,
        'direction_accuracy': dir_accuracy,
        'comparison_df': comparison_df,
        'test_size': len(X_test),
    }


def get_stock_summary(data):
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    change = latest['Close'] - prev['Close']
    change_pct = (change / prev['Close']) * 100

    return {
        'current_price': latest['Close'], 'open': latest['Open'],
        'high': latest['High'], 'low': latest['Low'],
        'close': latest['Close'], 'volume': latest['Volume'],
        'change': change, 'change_pct': change_pct,
        'high_52w': data['Close'].tail(252).max(),
        'low_52w': data['Close'].tail(252).min(),
        'avg_volume_30d': data['Volume'].tail(30).mean(),
        'last_date': latest['Date'] if 'Date' in data.columns else 'N/A',
    }
