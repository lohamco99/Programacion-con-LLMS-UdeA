import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def normalizar_por_ventana(df, target_col, ventana_horas, freq_min):
    df_work = df.copy()

    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work.index = pd.to_datetime(df_work.index)

    df_work = df_work.sort_index()

    # Regularizar la frecuencia temporal solicitada
    df_work = df_work.asfreq(f"{freq_min}min")

    feature_cols = [col for col in df_work.columns if col != target_col]

    # Interpolación de características, no del target
    df_work[feature_cols] = (
        df_work[feature_cols]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )

    X_parts = []
    y_parts = []

    for _, group in df_work.resample(f"{ventana_horas}h"):
        if len(group) == 0:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(group[feature_cols].to_numpy())

        X_parts.append(X_scaled)
        y_parts.append(group[target_col].to_numpy())

    if len(X_parts) == 0:
        return np.empty((0, len(feature_cols))), np.array([])

    X_procesado = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    return X_procesado, y
