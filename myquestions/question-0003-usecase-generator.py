import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score

def generar_caso_de_uso_detectar_adulteracion_aceite():
    """
    Genera casos para clasificación calibrada (Sklearn).
    """
    X = np.random.rand(500, 5)
    y = np.random.randint(0, 2, 500)
    method = np.random.choice(["sigmoid", "isotonic"])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
    
    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    cal = CalibratedClassifierCV(rf, method=method, cv=3).fit(X_tr_s, y_tr)
    
    probs = cal.predict_proba(X_te_s)[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    objeto_esperado = {
        "modelo_calibrado": cal,
        "roc_auc": round(float(roc_auc_score(y_te, probs)), 4),
        "brier_score": round(float(brier_score_loss(y_te, probs)), 4),
        "recall_positivo": round(float(recall_score(y_te, y_pred)), 4),
    }

    argumentos_entrada = {"X": X, "y": y, "test_size": 0.25, "calibration_method": method}
    return (argumentos_entrada, objeto_esperado)
