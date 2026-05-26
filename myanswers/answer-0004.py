import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering


def segmentar_zonas_sismicas(df, n_clusters, n_components_pca):
    df_num = df.select_dtypes(include=[np.number]).copy()
    numeric_cols = df_num.columns

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(df_num)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)

    n_components = min(n_components_pca, X_scaled.shape[1])

    pca = PCA(
        n_components=n_components,
        random_state=42
    )
    X_pca = pca.fit_transform(X_scaled)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )

    labels = clustering.fit_predict(X_pca)

    df_res = pd.DataFrame(X_imp, columns=numeric_cols)
    df_res["_cluster"] = labels

    resumen_clusters = (
        df_res
        .groupby("_cluster")
        .median()
        .reset_index(drop=True)
    )

    resumen_clusters["n_eventos"] = (
        df_res
        .groupby("_cluster")
        .size()
        .values
    )

    return labels, resumen_clusters
