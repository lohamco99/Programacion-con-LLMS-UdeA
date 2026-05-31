import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_poisson_deviance

def clasificacion_umbral_personalizado(df, target_col, umbral):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    nb = GaussianNB()
    nb.fit(X, y)

    probabilidades = nb.predict_proba(X)[:, 1]

    predicciones = np.where(probabilidades >= umbral, 1, 0)

    return probabilidades, predicciones
