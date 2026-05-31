import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_poisson_deviance

def calcular_devianza_aerogel(df_materiales, target_name):
    df = df_materiales.copy()

    X = df.drop(columns=[target_name]).values
    y = df[target_name].values

    y = np.where(y <= 0, 0.1, y)

    model = Ridge(alpha=2.0, random_state=123)
    model.fit(X, y)

    preds = model.predict(X)
    preds = np.clip(preds, 0.001, None)

    dev = float(mean_poisson_deviance(y, preds))

    resultado = int(dev * 1000)

    return resultado
