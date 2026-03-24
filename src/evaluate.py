import numpy as np

def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    res = np.zeros_like(y_true, dtype=float)
    res[mask] = np.abs(y_pred[mask] - y_true[mask]) / denom[mask]
    return float(np.mean(res)) * 100.0