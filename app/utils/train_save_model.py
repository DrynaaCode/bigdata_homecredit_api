import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Chargement des données
df = pd.read_csv("data/application_train.csv")

# Liste des features à utiliser (hors SK_ID_CURR et TARGET)
features = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED", "OWN_CAR_AGE",
    "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_CONT_MOBILE", "FLAG_EMAIL", "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "HOUSETYPE_MODE", "TOTALAREA_MODE",
    "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"
]

# Prétraitement minimal (remplacement des valeurs manquantes)
X = df[features].fillna(0)
y = df["TARGET"]

# Encodage des variables catégorielles
X = pd.get_dummies(X)

# Entraînement du modèle
model = RandomForestClassifier()
model.fit(X, y)

# Sauvegarde du modèle
with open("models_artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle entraîné et sauvegardé avec toutes les features.")