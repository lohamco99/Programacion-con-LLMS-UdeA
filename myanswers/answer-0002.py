import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

def predecir_ciclo_asimetrico(df, target_col, peso_subestimacion):
    """
    Predice el tiempo de ciclo penalizando más la subestimación.
    """
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    # Split con random_state fijo para consistencia con el generador
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    
    # Cálculo de error asimétrico
    subestimacion = y_pred < y_test
    errores = np.where(subestimacion, 
                       peso_subestimacion * (y_test - y_pred)**2, 
                       (y_pred - y_test)**2)
    
    return {
        "modelo": model,
        "wmse": round(float(errores.mean()), 4),
        "n_subestimaciones": int(subestimacion.sum())
    }
