import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


def generar_caso_de_uso_predecir_ciclo_asimetrico():
    """
    Genera un caso de uso aleatorio para:
    predecir_ciclo_asimetrico(df, target_col, peso_subestimacion)

    Retorna:
        argumentos_entrada (dict)
        objeto_esperado (dict)
    """
    rng = np.random.default_rng()

    n = int(rng.integers(300, 501))
    peso_subestimacion = round(float(rng.uniform(1.5, 5.0)), 2)

    # Variables explicativas sintéticas
    v1 = rng.uniform(0, 1, n)
    v2 = rng.uniform(10, 100, n)
    v3 = rng.normal(0, 1, n)
    v4 = rng.uniform(-5, 5, n)

    # Target continuo con relación no trivial
    ruido = rng.normal(0, 4, n)
    tiempo_ciclo = 50 + 18 * v1 + 0.35 * v2 - 6 * v3 + 2.5 * v4 + ruido

    df = pd.DataFrame({
        "sensor_A": v1,
        "sensor_B": v2,
        "sensor_C": v3,
        "sensor_D": v4,
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
        peso_subestimacion * (y_te - y_pred) ** 2,
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
        "peso_subestimacion": peso_subestimacion
    }

    return argumentos_entrada, objeto_esperado
