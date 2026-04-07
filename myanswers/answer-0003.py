import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score

def detectar_adulteracion_aceite(X, y, test_size, calibration_method):
    """
    Detecta fraude con probabilidades calibradas.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Modelo con pesos balanceados y calibración
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    calibrated_model = CalibratedClassifierCV(rf, method=calibration_method, cv=3)
    calibrated_model.fit(X_train_s, y_train)
    
    probs = calibrated_model.predict_proba(X_test_s)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    return {
        "modelo_calibrado": calibrated_model,
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probs)), 4),
        "recall_positivo": round(float(recall_score(y_test, preds)), 4)
    }
