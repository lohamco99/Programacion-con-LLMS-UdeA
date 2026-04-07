import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def generar_caso_de_uso_segmentar_zonas_sismicas():
    """
    Genera casos para clustering jerárquico (Sklearn y Pandas).
    """
    n = np.random.randint(300, 500)
    df = pd.DataFrame(np.random.rand(n, 4), columns=["mag", "prof", "dist", "acc"])
    n_cl = np.random.randint(2, 5)
    n_pca = 2

    # Inyección de nulos para probar imputer
    df.iloc[0:5, 0] = np.nan

    X_imp = SimpleImputer(strategy="median").fit_transform(df)
    X_sc = RobustScaler().fit_transform(X_imp)
    X_pca = PCA(n_components=n_pca, random_state=42).fit_transform(X_sc)
    labels = AgglomerativeClustering(n_clusters=n_cl).fit_predict(X_pca)
    
    df_res = pd.DataFrame(X_imp, columns=df.columns)
    df_res["_cluster"] = labels
    resumen = df_res.groupby("_cluster").median().reset_index(drop=True)
    resumen["n_eventos"] = df_res.groupby("_cluster").size().values

    objeto_esperado = (labels, resumen)
    argumentos_entrada = {"df": df, "n_clusters": n_cl, "n_components_pca": n_pca}
    return (argumentos_entrada, objeto_esperado)
