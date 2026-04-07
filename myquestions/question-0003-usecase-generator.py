import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score


def casos_de_uso_aleatorios_reto3():
    np.random.seed(np.random.randint(0, 9999))

    n           = np.random.randint(400, 800)
    n_features  = np.random.randint(8, 20)
    fraud_ratio = round(float(np.random.uniform(0.10, 0.20)), 2)
    test_size   = round(float(np.random.choice([0.20, 0.25, 0.30])), 2)
    method      = str(np.random.choice(["sigmoid", "isotonic"]))

    n_fraud = int(n * fraud_ratio)
    n_pure  = n - n_fraud
    X = np.vstack([
        np.random.normal(0.0, 1.0, (n_pure,  n_features)),
        np.random.normal(0.6, 1.2, (n_fraud, n_features)),
    ])
    y = np.array([0] * n_pure + [1] * n_fraud)

    idx  = np.random.permutation(n)
    X, y = X[idx], y[idx]

    # Output esperado
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    base = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
    cal = CalibratedClassifierCV(estimator=base, method=method, cv=3)
    cal.fit(X_tr_s, y_tr)

    proba  = cal.predict_proba(X_te_s)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    return {
        "input": {
            "X":                  X,
            "y":                  y,
            "test_size":          test_size,
            "calibration_method": method,
        },
        "output": {
            "roc_auc":         round(float(roc_auc_score(y_te, proba)), 4),
            "brier_score":     round(float(brier_score_loss(y_te, proba)), 4),
            "recall_positivo": round(float(recall_score(y_te, y_pred)), 4),
        },
    }
