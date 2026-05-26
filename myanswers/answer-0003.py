import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score


def detectar_adulteracion_aceite(X, y, test_size, calibration_method):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    )

    modelo_calibrado = CalibratedClassifierCV(
        rf,
        method=calibration_method,
        cv=3
    )

    modelo_calibrado.fit(X_tr_s, y_tr)

    probs = modelo_calibrado.predict_proba(X_te_s)[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    return {
        "modelo_calibrado": modelo_calibrado,
        "roc_auc": round(float(roc_auc_score(y_te, probs)), 4),
        "brier_score": round(float(brier_score_loss(y_te, probs)), 4),
        "recall_positivo": round(float(recall_score(y_te, y_pred)), 4)
    }
