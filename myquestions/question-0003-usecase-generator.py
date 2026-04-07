import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score

def detectar_adulteracion_aceite(X, y, test_size, calibration_method):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
    model = CalibratedClassifierCV(RandomForestClassifier(class_weight="balanced", random_state=42), method=calibration_method, cv=3).fit(X_tr_s, y_tr)
    probs = model.predict_proba(X_te_s)[:, 1]
    return {"roc_auc": round(roc_auc_score(y_te, probs), 4), "brier_score": round(brier_score_loss(y_te, probs), 4)}

def casos_de_uso_aleatorios():
    X, y = np.random.rand(500, 10), np.random.randint(0, 2, 500)
    return {"input": {"X": X, "y": y, "test_size": 0.25, "calibration_method": "sigmoid"}, "output": ["roc_auc"]}
