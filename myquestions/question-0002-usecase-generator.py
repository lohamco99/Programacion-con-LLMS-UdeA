import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

def generar_caso_de_uso_predecir_ciclo_asimetrico():
    """
    Genera casos para regresión asimétrica (Pandas y Sklearn).
    """
    n = np.random.randint(300, 500)
    peso = round(np.random.uniform(1.5, 5.0), 1)
    
    df = pd.DataFrame({
        "v1": np.random.rand(n), 
        "v2": np.random.rand(n),
        "tiempo_ciclo": np.random.uniform(100, 200, n)
    })

    X = df.drop(columns=["tiempo_ciclo"]).values
    y = df["tiempo_ciclo"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    
    model = GradientBoostingRegressor(random_state=42).fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    
    sub = y_pred < y_te
    err = np.where(sub, peso * (y_te - y_pred)**2, (y_pred - y_test)**2)
    
    objeto_esperado = {
        "modelo": model,
        "wmse": round(float(err.mean()), 4),
        "n_subestimaciones": int(sub.sum())
    }

    argumentos_entrada = {"df": df, "target_col": "tiempo_ciclo", "peso_subestimacion": peso}
    return (argumentos_entrada, objeto_esperado)
