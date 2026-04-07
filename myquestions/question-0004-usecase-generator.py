import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


def generar_caso_de_uso_segmentar_zonas_sismicas():
    """
    Genera un caso de uso aleatorio para:
    segmentar_zonas_sismicas(df, n_clusters, n_components_pca)

    Retorna:
        argumentos_entrada (dict)
        objeto_esperado (tuple): (labels, resumen_clusters)
    """
    rng = np.random.default_rng()

    n = int(rng.integers(300, 501))

    df = pd.DataFrame({
        "magnitud": rng.normal(4.8, 0.9, n),
        "profundidad": rng.normal(35, 12, n),
        "dist_epicentral": rng.normal(120, 45, n),
        "freq_dominante": rng.normal(6.2, 1.1, n),
        "duracion": rng.normal(18, 5, n),
        "pico_aceleracion": rng.normal(0.35, 0.12, n),
        "estacion": rng.choice(["A", "B", "C", "D"], size=n)  # no numérica
    })

    # Inyectar NaN aleatorios en columnas numéricas
    columnas_numericas_base = [
        "magnitud",
        "profundidad",
        "dist_epicentral",
        "freq_dominante",
        "duracion",
        "pico_aceleracion"
    ]

    for col in columnas_numericas_base:
        mask = rng.random(n) < 0.06
        df.loc[mask, col] = np.nan

    # Inyectar algunos outliers
    idx_outliers = rng.choice(n, size=max(3, n // 50), replace=False)
    df.loc[idx_outliers, "pico_aceleracion"] = (
        df.loc[idx_outliers, "pico_aceleracion"].fillna(0.35) * rng.uniform(5, 12)
    )
    df.loc[idx_outliers, "magnitud"] = (
        df.loc[idx_outliers, "magnitud"].fillna(4.8) + rng.uniform(2, 4)
    )

    n_clusters = int(rng.integers(2, 6))
    n_components_pca = int(rng.integers(2, len(columnas_numericas_base) + 3))

    # 1. Seleccionar solo numéricas
    df_num = df.select_dtypes(include=np.number)

    # 2. Imputación por mediana
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df_num)

    # 3. Escalado robusto
    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X_imp)

    # 4. PCA con min(n_components_pca, n_cols)
    n_cols = df_num.shape[1]
    n_comp_efectivo = min(n_components_pca, n_cols)

    pca = PCA(n_components=n_comp_efectivo)
    X_pca = pca.fit_transform(X_sc)

    # 5. Clustering jerárquico con linkage='ward'
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )
    labels = clustering.fit_predict(X_pca)

    # 6. Resumen por cluster usando datos imputados antes de escalar
    df_imp = pd.DataFrame(X_imp, columns=df_num.columns)
    df_imp["_cluster"] = labels

    resumen_clusters = df_imp.groupby("_cluster").median()
    resumen_clusters["n_eventos"] = df_imp.groupby("_cluster").size()

    # Índice reiniciado desde 0
    resumen_clusters = resumen_clusters.reset_index(drop=True)

    objeto_esperado = (labels, resumen_clusters)

    argumentos_entrada = {
        "df": df,
        "n_clusters": n_clusters,
        "n_components_pca": n_components_pca
    }

    return argumentos_entrada, objeto_esperado
