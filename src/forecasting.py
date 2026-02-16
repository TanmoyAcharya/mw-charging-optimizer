import numpy as np
from sklearn.ensemble import RandomForestRegressor

def make_features(base_load_kw: np.ndarray, lags: int = 8):
    """
    Build simple lag features: base[t-1], base[t-2], ... base[t-lags]
    """
    T = len(base_load_kw)
    X, y = [], []
    for t in range(lags, T):
        X.append(base_load_kw[t-lags:t][::-1])  # most recent first
        y.append(base_load_kw[t])
    return np.asarray(X), np.asarray(y)

def train_forecaster(base_load_kw: np.ndarray, lags: int = 8):
    X, y = make_features(base_load_kw, lags=lags)
    split = int(0.8 * len(X))
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]

    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(Xtr, ytr)

    # simple eval
    preds = model.predict(Xte)
    mae = float(np.mean(np.abs(preds - yte)))
    return model, mae, lags

def forecast_next(base_load_kw: np.ndarray, model, lags: int, horizon: int = 12):
    """
    Recursive forecast for next `horizon` steps.
    """
    hist = base_load_kw.copy().astype(float).tolist()
    out = []
    for _ in range(horizon):
        x = np.array(hist[-lags:][::-1]).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        out.append(yhat)
        hist.append(yhat)
    return np.array(out)
