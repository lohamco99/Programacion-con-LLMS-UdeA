import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


def generar_caso_de_uso_predecir_ciclo_asimetrico():
    """
    Genera un caso de uso aleatorio para predecir_ciclo_asimetrico(df, target_col, peso_subestimacion).
    Devuelve:
        - argumentos_entrada: dict
        - objeto_esperado: dict
    """
    rng = np.random.default_rng()

    n = int(rng.integers(300, 500))
    peso = round(float(rng.uniform(1.5, 5.0)), 1)

    # Datos sintéticos con relación real entre variables y target
    v1 = rng.uniform(0, 1, n)
    v2 = rng.uniform(0, 1, n)
    v3 = rng.normal(0, 1, n)
    ruido = rng.normal(0, 3, n)

    tiempo_ciclo = 120 + 25 * v1 - 18 * v2 + 7 * v3 + ruido

    df = pd.DataFrame({
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "tiempo_ciclo": tiempo_ciclo
    })

    target_col = "tiempo_ciclo"

    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
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
        peso * (y_te - y_pred) ** 2,
        (y_pred - y_te) ** 2
    )

    objeto_esperado = {
        "modelo": modelo,
        "wmse": round(float(np.mean(errores)), 4),
        "n_subestimaciones": int(np.sum(subestimaciones))
    }

    argumentos_entrada = {
        "df": df,
        "target_col": target_col,
        "peso_subestimacion": peso
    }

    return argumentos_entrada, objeto_esperado
