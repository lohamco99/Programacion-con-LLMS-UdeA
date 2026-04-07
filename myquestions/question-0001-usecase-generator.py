import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalizar_por_ventana(df, target_col, ventana_horas, freq_min):
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    feature_cols = [c for c in df.columns if c != target_col]
    df[feature_cols] = df[feature_cols].interpolate(method="linear").bfill().ffill()
    ventana_str = f"{ventana_horas}h"
    scaler = StandardScaler()
    X_parts, y_parts = [], []
    for _, group in df.resample(ventana_str):
        if len(group) < 1: continue
        X_w = group[feature_cols].values
        y_w = group[target_col].values
        X_scaled = scaler.fit_transform(X_w)
        X_parts.append(X_scaled)
        y_parts.append(y_w)
    return np.vstack(X_parts), np.concatenate(y_parts)

def casos_de_uso_aleatorios():
    np.random.seed(np.random.randint(0, 9999))
    n, freq, ventana = np.random.randint(200, 400), np.random.choice([5, 10]), np.random.choice([2, 4])
    idx = pd.date_range("2024-01-01", periods=n, freq=f"{freq}min")
    df = pd.DataFrame(np.random.rand(n, 4), columns=["f1", "f2", "f3", "target"], index=idx)
    df.iloc[0:5, 0:2] = np.nan
    return {"input": {"df": df, "target_col": "target", "ventana_horas": ventana, "freq_min": freq}, "output": ["X_shape", "y_shape"]}
