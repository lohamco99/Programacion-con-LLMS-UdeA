import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_poisson_deviance

def calcular_costo_total_por_material(df):
    df = df.copy()

    df["costo_total"] = df["costo_unitario"] * df["cantidad"]

    resultado = df.groupby("material")["costo_total"].sum().to_dict()

    return resultado
