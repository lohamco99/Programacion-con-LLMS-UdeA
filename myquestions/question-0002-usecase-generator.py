import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

def predecir_ciclo_asimetrico(df, target_col, peso_subestimacion):
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train_s, X_test_s = sc.fit_transform(X_train), sc.transform(X_test)
    model = GradientBoostingRegressor(random_state=42).fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    sub = y_pred < y_test
    err = np.where(sub, peso_subestimacion * (y_test - y_pred)**2, (y_pred - y_test)**2)
    return {"modelo": model, "wmse": round(float(err.mean()), 4), "n_subestimaciones": int(sub.sum())}

def casos_de_uso_aleatorios():
    n, peso = 400, 3.5
    df = pd.DataFrame(np.random.rand(n, 5), columns=["v1", "v2", "v3", "v4", "target"])
    return {"input": {"df": df, "target_col": "target", "peso_subestimacion": peso}, "output": ["wmse"]}
