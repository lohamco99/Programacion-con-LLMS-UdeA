import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


def casos_de_uso_aleatorios_reto4():
    np.random.seed(np.random.randint(0, 9999))

    n          = np.random.randint(300, 700)
    n_clusters = int(np.random.randint(3, 6))
    n_comp     = int(np.random.randint(2, 5))

    df = pd.DataFrame({
        "magnitud":       np.random.exponential(2.5, n) + 1.0,
        "profundidad_km": np.random.uniform(5, 200, n),
        "distancia_km":   np.abs(np.random.normal(80, 40, n)),
        "frec_dom_hz":    np.random.uniform(0.5, 15.0, n),
        "duracion_seg":   np.random.exponential(30, n) + 5,
        "pga_gal":        np.random.lognormal(2.0, 1.0, n),
        "pgv_cms":        np.random.lognormal(1.0, 0.8, n),
        "ratio_sp":       np.random.uniform(1.0, 8.0, n),
        "energia_joules": np.random.lognormal(10, 2, n),
        "azimut_grados":  np.random.uniform(0, 360, n),
        "snr_db":         np.random.normal(15, 5, n),
        "num_fases":      np.random.randint(3, 20, n).astype(float),
    })

    for col in df.columns:
        df.loc[np.random.rand(n) < 0.06, col] = np.nan
    df.loc[np.random.choice(n, 5, replace=False), "pga_gal"] *= 50

    # Output esperado
    df_num       = df.select_dtypes(include="number")
    feature_cols = df_num.columns.tolist()

    X_imp = SimpleImputer(strategy="median").fit_transform(df_num)
    X_sc  = RobustScaler().fit_transform(X_imp)

    n_comp_real = min(n_comp, X_sc.shape[1])
    X_pca  = PCA(n_components=n_comp_real, random_state=42).fit_transform(X_sc)
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward"
    ).fit_predict(X_pca)

    df_orig          = pd.DataFrame(X_imp, columns=feature_cols)
    df_orig["_clus"] = labels
    resumen = (
        df_orig.groupby("_clus")[feature_cols]
               .median()
               .reset_index(drop=True)
    )
    resumen["n_eventos"] = df_orig.groupby("_clus").size().values

    return {
        "input": {
            "df":               df,
            "n_clusters":       n_clusters,
            "n_components_pca": n_comp,
        },
        "output": {
            "labels_shape":           labels.shape,
            "n_clusters_encontrados": int(len(np.unique(labels))),
            "resumen_shape":          resumen.shape,
            "resumen_columnas":       list(resumen.columns),
        },
    }
