import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def generar_caso_de_uso_normalizar_por_ventana():
    n         = np.random.randint(200, 400)
    freq_min  = int(np.random.choice([5, 10, 15]))
    ventana_h = int(np.random.choice([2, 4, 6]))

    idx = pd.date_range("2024-01-01", periods=n, freq=f"{freq_min}min")
    df = pd.DataFrame({
        "temperatura": np.random.normal(37.0, 1.5, n),
        "ph":          np.random.normal(7.0,  0.3, n),
        "oxigeno":     np.random.normal(8.5,  0.8, n),
        "rendimiento": np.random.uniform(70, 99, n),
    }, index=idx)

    for col in ["temperatura", "ph", "oxigeno"]:
        df.loc[np.random.rand(n) < 0.08, col] = np.nan

    # Output esperado
    feature_cols = ["temperatura", "ph", "oxigeno"]
    df_clean = df.copy()
    df_clean[feature_cols] = (
        df_clean[feature_cols]
        .interpolate(method="linear")
        .bfill().ffill()
    )

    scaler  = StandardScaler()
    X_parts, y_parts = [], []
    for _, group in df_clean.resample(f"{ventana_h}h"):
        if len(group) == 0:
            continue
        X_parts.append(scaler.fit_transform(group[feature_cols].values))
        y_parts.append(group["rendimiento"].values)

    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)

    return {
        "input": {
            "df":            df,
            "target_col":    "rendimiento",
            "ventana_horas": ventana_h,
            "freq_min":      freq_min,
        },
        "output": {
            "X_shape":            X_out.shape,
            "y_shape":            y_out.shape,
            "X_mean_approx_zero": bool(np.abs(X_out.mean()) < 1.0),
            "no_nan_in_X":        bool(not np.isnan(X_out).any()),
        },
    }
