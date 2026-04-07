import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


def casos_de_uso_aleatorios_reto2():
    np.random.seed(np.random.randint(0, 9999))

    n    = np.random.randint(300, 600)
    peso = round(float(np.random.uniform(1.5, 5.0)), 1)

    df = pd.DataFrame({
        "velocidad_husillo": np.random.uniform(500,  3000, n),
        "profundidad_corte": np.random.uniform(0.1,   5.0, n),
        "avance_mm_min":     np.random.uniform(50,    500, n),
        "temperatura_herr":  np.random.normal(45,       8, n),
        "vibracion_rms":     np.abs(np.random.normal(0.3, 0.1, n)),
        "tiempo_ciclo_seg":  120 + np.random.uniform(0, 80, n)
                             + np.random.normal(0, 10, n),
    })

    # Output esperado
    X = df.drop(columns=["tiempo_ciclo_seg"]).values
    y = df["tiempo_ciclo_seg"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    model  = GradientBoostingRegressor(random_state=42)
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)

    subestim = y_pred < y_te
    errores  = np.where(
        subestim,
        peso * (y_te - y_pred) ** 2,
        (y_pred - y_te) ** 2,
    )

    return {
        "input": {
            "df":                 df,
            "target_col":         "tiempo_ciclo_seg",
            "peso_subestimacion": peso,
        },
        "output": {
            "wmse":              round(float(errores.mean()), 4),
            "n_subestimaciones": int(subestim.sum()),
        },
    }
