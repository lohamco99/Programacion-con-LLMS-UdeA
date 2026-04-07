import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


def generar_caso_de_uso_segmentar_zonas_sismicas():
    """
    Genera un caso de uso aleatorio para segmentar_zonas_sismicas(df, n_clusters, n_components_pca).
    Devuelve:
        - argumentos_entrada: dict
        - objeto_esperado: tuple(labels, resumen_clusters)
    """
    rng = np.random.default_rng()

    n = int(rng.integers(300, 501))

    df = pd.DataFrame({
        "mag": rng.normal(4.5, 0.9, n),
        "prof": rng.normal(35, 12, n),
        "dist": rng.normal(120, 40, n),
        "freq_dom": rng.normal(6.0, 1.2, n),
        "duracion": rng.normal(18, 5, n),
        "acc": rng.normal(0.35, 0.12, n),
        "estacion": rng.choice(["A", "B", "C", "D"], size=n)  # no numérica
    })

    # Inyectar NaN aleatorios en columnas numéricas
    num_cols = ["mag", "prof", "dist", "freq_dom", "duracion", "acc"]
    for col in num_cols:
        mask = rng.random(n) < 0.06
        df.loc[mask, col] = np.nan

    # Inyectar algunos outliers
    idx_outliers = rng.choice(n, size=max(3, n // 50), replace=False)
    df.loc[idx_outliers, "acc"] = df.loc[idx_outliers, "acc"].fillna(0.35) * rng.uniform(5, 12)
    df.loc[idx_outliers, "mag"] = df.loc[idx_outliers, "mag"].fillna(4.5) + rng.uniform(2, 4)

    n_clusters = int(rng.integers(2, 6))
    n_components_pca = int(rng.integers(2, len(num_cols) + 3))  # a veces mayor que n_cols

    df_num = df.select_dtypes(include=np.number)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df_num)

    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X_imp)

    n_comp_efectivo = min(n_components_pca, df_num.shape[1])
    X_pca = PCA(n_components=n_comp_efectivo).fit_transform(X_sc)

    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    ).fit_predict(X_pca)

    df_imp = pd.DataFrame(X_imp, columns=df_num.columns)
    df_imp["_cluster"] = labels

    resumen_clusters = df_imp.groupby("_cluster").median()
    resumen_clusters["n_eventos"] = df_imp.groupby("_cluster").size()
    resumen_clusters = resumen_clusters.reset_index(drop=True)

    objeto_esperado = (labels, resumen_clusters)

    argumentos_entrada = {
        "df": df,
        "n_clusters": n_clusters,
        "n_components_pca": n_components_pca
    }

    return argumentos_entrada, objeto_esperado
