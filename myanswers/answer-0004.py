import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def segmentar_zonas_sismicas(df, n_clusters, n_components_pca):
    """
    Segmenta zonas mediante clustering jerárquico y reducción PCA.
    """
    df_num = df.select_dtypes(include="number")
    cols = df_num.columns
    
    # Imputación y Escalamiento Robusto
    X_imp = SimpleImputer(strategy="median").fit_transform(df_num)
    X_sc = RobustScaler().fit_transform(X_imp)
    
    # PCA para reducción de ruido
    pca = PCA(n_components=n_components_pca, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    
    # Clustering Jerárquico (Ward)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(X_pca)
    
    # Crear tabla de resumen con medianas originales
    df_res = pd.DataFrame(X_imp, columns=cols)
    df_res["_cluster"] = labels
    resumen = df_res.groupby("_cluster").median().reset_index(drop=True)
    resumen["n_eventos"] = df_res.groupby("_cluster").size().values
    
    return labels, resumen
