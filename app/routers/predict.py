from fastapi import APIRouter, HTTPException
from app.services.database import get_database
import os
import pickle
import numpy as np

router = APIRouter()

@router.get("/predict/{id}")
async def predict_client(id: str):
    """
    Prédit le score d'un client à partir de son SK_ID_CURR.
    """
    db = await get_database()
    collection_name = os.getenv("MONGO_COLL_CLIENT", "client")
    try:
        sk_id = int(id)
    except ValueError:
        raise HTTPException(status_code=400, detail="SK_ID_CURR doit être un entier")
    client = await db[collection_name].find_one({"SK_ID_CURR": sk_id})
    if not client:
        raise HTTPException(status_code=404, detail="Client non trouvé")

    # Charger le modèle pkl
    model_path = os.getenv("MODEL_PATH", "models_artifacts/model.pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement modèle: {e}")

    # Préparer les features du client (adapter selon ton modèle)
    try:
        import pandas as pd
        # Liste des features utilisées à l'entraînement
        features_list = [
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
        # Crée un DataFrame avec les features du client
        client_features = {k: client.get(k, 0) for k in features_list}
        df_client = pd.DataFrame([client_features])
        # Encodage des variables catégorielles comme à l'entraînement
        df_client = pd.get_dummies(df_client)
        # Aligne les colonnes avec celles du modèle
        model_features = getattr(model, 'feature_names_in_', None)
        if model_features is not None:
            for col in model_features:
                if col not in df_client.columns:
                    df_client[col] = 0
            df_client = df_client[model_features]
        features = df_client.values
        score = float(model.predict_proba(features)[0][1])  # proba défaut
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {e}")

    return {"id": id, "score": score}