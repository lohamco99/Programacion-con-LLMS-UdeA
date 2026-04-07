import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score


def generar_caso_de_uso_detectar_adulteracion_aceite():
    """
    Genera un caso de uso aleatorio para detectar_adulteracion_aceite(X, y, test_size, calibration_method).
    Devuelve:
        - argumentos_entrada: dict
        - objeto_esperado: dict
    """
    rng = np.random.default_rng()

    n_samples = int(rng.integers(400, 801))
    n_features = int(rng.integers(5, 11))
    test_size = round(float(rng.uniform(0.1, 0.4)), 2)
    calibration_method = rng.choice(["sigmoid", "isotonic"]).item()

    # Aproximadamente 15% de positivos, como pide el enunciado
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(3, n_features // 2),
        n_redundant=max(1, n_features // 4),
        n_repeated=0,
        n_classes=2,
        weights=[0.85, 0.15],
        class_sep=1.2,
        random_state=42
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
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

    objeto_esperado = {
        "modelo_calibrado": modelo_calibrado,
        "roc_auc": round(float(roc_auc_score(y_te, probs)), 4),
        "brier_score": round(float(brier_score_loss(y_te, probs)), 4),
        "recall_positivo": round(float(recall_score(y_te, y_pred)), 4),
    }

    argumentos_entrada = {
        "X": X,
        "y": y,
        "test_size": test_size,
        "calibration_method": calibration_method
    }

    return argumentos_entrada, objeto_esperado
