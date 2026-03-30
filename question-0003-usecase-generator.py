import numpy as np

def casos_de_uso_aleatorios_reto3():
    np.random.seed(np.random.randint(0, 9999))
    n = np.random.randint(400, 800)
    n_features = np.random.randint(8, 20)
    fraud_ratio = round(np.random.uniform(0.10, 0.20), 2)
    method = np.random.choice(["sigmoid", "isotonic"])

    # Simular espectros NIR: puras vs adulteradas con shift
    n_fraud = int(n * fraud_ratio)
    n_pure  = n - n_fraud
    X_pure  = np.random.normal(0.0, 1.0, (n_pure, n_features))
    X_fraud = np.random.normal(0.6, 1.2, (n_fraud, n_features))  # distribución diferente
    X = np.vstack([X_pure, X_fraud])
    y = np.array([0] * n_pure + [1] * n_fraud)

    # Shuffle
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    base = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    cal = CalibratedClassifierCV(base, method=method, cv=3)
    cal.fit(X_tr_s, y_tr)
    proba = cal.predict_proba(X_te_s)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    return {
        "input": {
            "X": X,
            "y": y,
            "test_size": 0.25,
            "calibration_method": method,
        },
        "output": {
            "roc_auc":        round(roc_auc_score(y_te, proba), 4),
            "brier_score":    round(brier_score_loss(y_te, proba), 4),
            "recall_positivo":round(recall_score(y_te, y_pred), 4),
        }
    }