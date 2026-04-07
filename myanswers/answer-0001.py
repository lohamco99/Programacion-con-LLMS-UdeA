import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalizar_por_ventana(df, target_col, ventana_horas, freq_min):
    """
    Limpia NaNs y normaliza por ventanas de tiempo para evitar el drift.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Limpieza de nulos
    df[feature_cols] = df[feature_cols].interpolate(method="linear").bfill().ffill()
    
    ventana_str = f"{ventana_horas}h"
    scaler = StandardScaler()
    X_parts, y_parts = [], []
    
    for _, group in df.resample(ventana_str):
        if len(group) < 1: continue
        
        X_w = group[feature_cols].values
        y_w = group[target_col].values
        
        # Fit y transform independiente por cada ventana
        X_scaled = scaler.fit_transform(X_w)
        X_parts.append(X_scaled)
        y_parts.append(y_w)
        
    return np.vstack(X_parts), np.concatenate(y_parts)
