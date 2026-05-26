import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


def predecir_ciclo_asimetrico(df, target_col, peso_subestimacion):
    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    modelo = GradientBoostingRegressor(random_state=42)
    modelo.fit(X_tr_s, y_tr)

    y_pred = modelo.predict(X_te_s)

    subestimaciones = y_pred < y_te

    errores = np.where(
        subestimaciones,
        peso_subestimacion * (y_te - y_pred) ** 2,
        (y_pred - y_te) ** 2
    )

    return {
        "modelo": modelo,
        "wmse": round(float(np.mean(errores)), 4),
        "n_subestimaciones": int(np.sum(subestimaciones))
    }
