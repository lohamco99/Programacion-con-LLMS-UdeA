import pandas as pd
import numpy as np

def casos_de_uso_aleatorios_reto4():
    np.random.seed(np.random.randint(0, 9999))
    n = np.random.randint(300, 700)
    n_clusters = np.random.randint(3, 6)
    n_comp = np.random.randint(2, 5)

    df = pd.DataFrame({
        "magnitud":       np.random.exponential(2.5, n) + 1.0,
        "profundidad_km": np.random.uniform(5, 200, n),
        "distancia_km":   np.abs(np.random.normal(80, 40, n)),
        "frec_dom_hz":    np.random.uniform(0.5, 15.0, n),
        "duracion_seg":   np.random.exponential(30, n) + 5,
        "pga_gal":        np.random.lognormal(2.0, 1.0, n),   # pico aceleración suelo
        "pgv_cms":        np.random.lognormal(1.0, 0.8, n),
        "ratio_sp":       np.random.uniform(1.0, 8.0, n),
        "energia_joules": np.random.lognormal(10, 2, n),
        "azimut_grados":  np.random.uniform(0, 360, n),
        "snr_db":         np.random.normal(15, 5, n),
        "num_fases":      np.random.randint(3, 20, n).astype(float),
    })

    # Inyectar NaN (~6%) y outliers extremos (~2%)
    for col in df.columns:
        mask_nan = np.random.rand(n) < 0.06
        df.loc[mask_nan, col] = np.nan
    df.loc[np.random.choice(n, 5, replace=False), "pga_gal"] *= 50  # outliers

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering

    df_num = df.select_dtypes(include="number")
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(df_num)
    sc = RobustScaler()
    X_sc = sc.fit_transform(X_imp)
    pca = PCA(n_components=min(n_comp, df_num.shape[1]))
    X_pca = pca.fit_transform(X_sc)
    clust = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = clust.fit_predict(X_pca)

    df_orig = pd.DataFrame(X_imp, columns=df_num.columns)
    df_orig["_cluster"] = labels
    resumen = (
        df_orig.groupby("_cluster")
               .agg(lambda x: np.median(x))
               .rename(columns={})
    )
    resumen["n_eventos"] = df_orig.groupby("_cluster").size()
    resumen = resumen.reset_index(drop=True)

    return {
        "input": {
            "df": df,
            "n_clusters": n_clusters,
            "n_components_pca": n_comp,
        },
        "output": {
            "labels_shape": labels.shape,
            "n_clusters_encontrados": len(np.unique(labels)),
            "resumen_shape": resumen.shape,
            "resumen_columnas": list(resumen.columns),
        }
    }