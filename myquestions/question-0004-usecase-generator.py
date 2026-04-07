import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def segmentar_zonas_sismicas(df, n_clusters, n_components_pca):
    df_num = df.select_dtypes(include="number")
    X_imp = SimpleImputer(strategy="median").fit_transform(df_num)
    X_pca = PCA(n_components=n_components_pca, random_state=42).fit_transform(RobustScaler().fit_transform(X_imp))
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_pca)
    df_res = pd.DataFrame(X_imp, columns=df_num.columns)
    df_res["_cluster"] = labels
    resumen = df_res.groupby("_cluster").median().reset_index(drop=True)
    resumen["n_eventos"] = df_res.groupby("_cluster").size().values
    return labels, resumen

def casos_de_uso_aleatorios():
    df = pd.DataFrame(np.random.rand(300, 6), columns=[f"s{i}" for i in range(6)])
    return {"input": {"df": df, "n_clusters": 3, "n_components_pca": 2}, "output": ["labels_shape"]}
