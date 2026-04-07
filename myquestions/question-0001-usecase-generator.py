import pandas as pd
import numpy as np

def casos_de_uso_aleatorios_reto1():
    np.random.seed(np.random.randint(0, 9999))
    n = np.random.randint(200, 400)
    freq = np.random.choice([5, 10, 15])
    ventana = np.random.choice([2, 4, 6])

    idx = pd.date_range("2024-01-01", periods=n, freq=f"{freq}min")
    df = pd.DataFrame({
        "temperatura": np.random.normal(37.0, 1.5, n),
        "ph":          np.random.normal(7.0, 0.3, n),
        "oxigeno":     np.random.normal(8.5, 0.8, n),
        "rendimiento": np.random.uniform(70, 99, n),  # target
    }, index=idx)

    # Inyectar NaN aleatorios (~8% de filas)
    for col in ["temperatura", "ph", "oxigeno"]:
        mask = np.random.rand(n) < 0.08
        df.loc[mask, col] = np.nan

    # Calcular output esperado (shape solamente, para validar)
    from sklearn.preprocessing import StandardScaler
    df_clean = df.copy()
    for col in ["temperatura", "ph", "oxigeno"]:
        df_clean[col] = df_clean[col].interpolate(method="linear").bfill().ffill()

    freq_str = f"{freq}min"
    ventana_str = f"{ventana}h"
    groups = df_clean.resample(ventana_str)
    X_parts, y_parts = [], []
    scaler = StandardScaler()
    for _, group in groups:
        if len(group) == 0:
            continue
        X_w = group.drop(columns=["rendimiento"]).values
        y_w = group["rendimiento"].values
        X_scaled = scaler.fit_transform(X_w)
        X_parts.append(X_scaled)
        y_parts.append(y_w)

    X_out = np.vstack(X_parts)
    y_out = np.concatenate(y_parts)

    return {
        "input": {
            "df": df,
            "target_col": "rendimiento",
            "ventana_horas": ventana,
            "freq_min": freq,
        },
        "output": {
            "X_shape": X_out.shape,
            "y_shape": y_out.shape,
            "X_mean_approx_zero": bool(np.abs(X_out.mean()) < 1.0),
        }
    }