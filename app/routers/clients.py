
# # Import des modules nécessaires
# from fastapi import APIRouter, HTTPException
# # Import de la fonction d'accès à la base MongoDB
# from app.services.database import get_database
# from app.schemas.clients  import ClientItem, ClientListResponse
# from app.schemas.clients import  ClientDetailResponse, ClientDetail, ClientRaw
# import os


# # Création du router pour regrouper les routes liées aux clients
# router = APIRouter()


# @router.get("/clients", response_model=ClientListResponse)
# async def get_clients(
#     skip: int = 0,
#     limit: int = 20,
#     gender: str = None,
#     contract_type: str = None,
#     income_type: str = None,
#     education_type: str = None,
#     family_status: str = None,
#     housing_type: str = None,
#     occupation_type: str = None,
#     min_income: float = None,
#     max_income: float = None,
#     min_credit: float = None,
#     max_credit: float = None,
#     min_annuity: float = None,
#     max_annuity: float = None,
#     min_goods_price: float = None,
#     max_goods_price: float = None,
#     min_children: int = None,
#     max_children: int = None,
#     min_fam_members: float = None,
#     max_fam_members: float = None,
#     region_rating: int = None,
#     target: int = None
# ):
#     """
#     Récupère une liste de clients paginée depuis la base MongoDB.
#     - skip : nombre de clients à ignorer (pour la pagination)
#     - limit : nombre maximum de clients à retourner
#     La réponse inclut aussi le nombre total, le nombre restant et les paramètres utilisés.
#     """
#     db = await get_database()  # Connexion à la base
#     collection_name = os.getenv("MONGO_COLL_CLIENTS", "client")  # Nom de la collection
#     try:
#         # Construction dynamique du filtre
#         query = {}
#         if gender:
#             query["CODE_GENDER"] = gender
#         if contract_type:
#             query["NAME_CONTRACT_TYPE"] = contract_type
#         if income_type:
#             query["NAME_INCOME_TYPE"] = income_type
#         if education_type:
#             query["NAME_EDUCATION_TYPE"] = education_type
#         if family_status:
#             query["NAME_FAMILY_STATUS"] = family_status
#         if housing_type:
#             query["NAME_HOUSING_TYPE"] = housing_type
#         if occupation_type:
#             query["OCCUPATION_TYPE"] = occupation_type
#         if region_rating:
#             query["REGION_RATING_CLIENT"] = region_rating
#         if target is not None:
#             query["TARGET"] = target

#         # Filtres numériques (intervalle)
#         def add_range_filter(field, min_val, max_val):
#             if min_val is not None or max_val is not None:
#                 query[field] = {}
#                 if min_val is not None:
#                     query[field]["$gte"] = min_val
#                 if max_val is not None:
#                     query[field]["$lte"] = max_val

#         add_range_filter("AMT_INCOME_TOTAL", min_income, max_income)
#         add_range_filter("AMT_CREDIT", min_credit, max_credit)
#         add_range_filter("AMT_ANNUITY", min_annuity, max_annuity)
#         add_range_filter("AMT_GOODS_PRICE", min_goods_price, max_goods_price)
#         add_range_filter("CNT_CHILDREN", min_children, max_children)
#         add_range_filter("CNT_FAM_MEMBERS", min_fam_members, max_fam_members)

#         # Récupération des clients avec pagination et filtre
#         cursor = db[collection_name].find(query).skip(skip).limit(limit)
#         clients = []

#         # Fonction utilitaire pour remplacer les NaN par None (JSON compatible)
#         import math
#         def replace_nan(obj):
#             if isinstance(obj, dict):
#                 return {k: replace_nan(v) for k, v in obj.items()}
#             elif isinstance(obj, list):
#                 return [replace_nan(v) for v in obj]
#             elif isinstance(obj, float) and math.isnan(obj):
#                 return None
#             else:
#                 return obj

#         # Parcours des résultats et conversion de l'_id + nettoyage des NaN
#         async for client in cursor:
#             # Contrôle du type de gender
#             raw_gender = client.get("CODE_GENDER")
#             if isinstance(raw_gender, str):
#                 gender = raw_gender
#             else:
#                 gender = None
#             item = ClientItem(
#                 id=str(client.get("SK_ID_CURR", "")),
#                 revenue=client.get("AMT_INCOME_TOTAL"),
#                 gender=gender
#             )
#             clients.append(item)

#         # Calcul du nombre total et du nombre restant
#         total_count = await db[collection_name].count_documents({})
#         pages = (total_count // limit) + (1 if total_count % limit else 0)
#         has_next = skip + limit < total_count
#         return ClientListResponse(
#             items=clients,
#             page=(skip // limit) + 1,
#             limit=limit,
#             total=total_count,
#             pages=pages,
#             has_next=has_next
#         )
#     except Exception as e:
#         # Gestion des erreurs : retourne un message d'erreur HTTP
#         raise HTTPException(status_code=400, detail=f"Erreur: {e}")



# # Route pour récupérer les infos d'un client par son SK_ID_CURR
# @router.get("/clients/{id}", response_model=ClientDetailResponse)
# async def get_client_by_id(id: str):
#     """
#     Récupère les infos détaillées d'un client par son identifiant SK_ID_CURR.
#     """
#     db = await get_database()
#     collection_name = os.getenv("MONGO_COLL_CLIENTS", "client")
#     try:
#         # Conversion sécurisée de l'id en int
#         try:
#             sk_id = int(id)
#         except ValueError:
#             raise HTTPException(status_code=400, detail="SK_ID_CURR doit être un entier")

#         client = await db[collection_name].find_one({"SK_ID_CURR": sk_id})
#         if not client:
#             raise HTTPException(status_code=404, detail="Client non trouvé")

#         # Retire le champ '_id' pour éviter l'erreur Pydantic
#         client.pop('_id', None)

#         # Nettoie les NaN dans le document client
#         import math
#         def replace_nan(obj):
#             if isinstance(obj, dict):
#                 return {k: replace_nan(v) for k, v in obj.items()}
#             elif isinstance(obj, list):
#                 return [replace_nan(v) for v in obj]
#             elif isinstance(obj, float) and math.isnan(obj):
#                 return None
#             else:
#                 return obj
#         client = replace_nan(client)

#         # Calcul du score du modèle pkl
#         import pickle
#         import numpy as np
#         import pandas as pd
#         model_path = os.getenv("MODEL_PATH", "models_artifacts/model.pkl")
#         try:
#             with open(model_path, "rb") as f:
#                 model = pickle.load(f)
#             features_list = [
#                 "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
#                 "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
#                 "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
#                 "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED", "OWN_CAR_AGE",
#                 "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_CONT_MOBILE", "FLAG_EMAIL", "OCCUPATION_TYPE",
#                 "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
#                 "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "HOUSETYPE_MODE", "TOTALAREA_MODE",
#                 "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE",
#                 "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
#                 "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
#                 "FLAG_DOCUMENT_COUNT"
#             ]
#             client_features = {k: client.get(k, 0) for k in features_list}
#             df_client = pd.DataFrame([client_features])
#             df_client = pd.get_dummies(df_client)
#             model_features = getattr(model, 'feature_names_in_', None)
#             if model_features is not None:
#                 for col in model_features:
#                     if col not in df_client.columns:
#                         df_client[col] = 0
#                 df_client = df_client[model_features]
#             features = df_client.values
#             score = float(model.predict_proba(features)[0][1])
#         except Exception as e:
#             score = 0.0  # Valeur par défaut en cas d'erreur

#         # Mapping vers ClientDetail
#         detail = ClientDetail(
#             id=str(client.get("SK_ID_CURR", "")),
#             revenue=client.get("AMT_INCOME_TOTAL"),
#             gender=client.get("CODE_GENDER"),
#             raw=ClientRaw.model_validate(client)
#         )
#         thr = float(os.getenv("THRESHOLD", "0.5"))
#         pred = 1 if score >= thr else 0
#         dec = "REJECT" if pred == 1 else "ACCEPT"
#         return ClientDetailResponse(
#             client=detail,
#             prediction=int(score),      # conversion explicite
#             proba_default=float(score), # conversion explicite
#             threshold=thr,
#             decision=dec,
#             top_features=[]
#         )
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Erreur: {e}")


# routers/clients.py
# -*- coding: utf-8 -*-

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Set
import os
import math
import numpy as np
import pickle
from functools import lru_cache

from app.services.database import get_database
from app.schemas.clients import (
    ClientItem,
    ClientListResponse,
    ClientDetailResponse,
    ClientDetail,
    ClientRaw,
)

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
router = APIRouter()

# -----------------------------------------------------------------------------
# Config / Modèle ML (chargé une seule fois)
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models_artifacts/model.pkl")

@lru_cache(maxsize=1)
def load_model():
    """Charge le modèle une seule fois en mémoire (évite re-pickle + OOM)."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# Champs catégoriels connus (qui ont été "one-hot" via get_dummies au training)
CATEGORICAL_BASES: Set[str] = {
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "HOUSETYPE_MODE",
}

# Champs numériques conservés tels quels au training
NUMERIC_FIELDS: Set[str] = {
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED", "OWN_CAR_AGE",
    "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_CONT_MOBILE", "FLAG_EMAIL",
    "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "TOTALAREA_MODE",
    "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "FLAG_DOCUMENT_COUNT",
}

def _to_float(v: Any) -> float:
    """Conversion robuste en float pour les champs numériques/flags."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        try:
            if isinstance(v, float) and math.isnan(v):
                return 0.0
            return float(v)
        except Exception:
            return 0.0
    # Flags textuels éventuels
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"y", "yes", "true", "1"}:
            return 1.0
        if s in {"n", "no", "false", "0"}:
            return 0.0
        # nombre sous forme de chaîne ?
        try:
            return float(v)
        except Exception:
            return 0.0
    return 0.0

def _replace_nan(obj: Any) -> Any:
    """Remplace les NaN par None pour JSON/pydantic."""
    if isinstance(obj, dict):
        return {k: _replace_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_nan(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

def to_feature_vector(client_doc: Dict[str, Any], model_features: np.ndarray) -> np.ndarray:
    """
    Construit un vecteur 1xN aligné sur model.feature_names_in_ sans pandas:
    - Les champs numériques sont repris bruts (convertis en float).
    - Les catégoriels sont encodés en one-hot "COL_valeur".
    """
    # Prépare les clés one-hot présentes pour ce client
    one_hot_keys = set()
    for base in CATEGORICAL_BASES:
        val = client_doc.get(base)
        if val is not None and val != "":
            one_hot_keys.add(f"{base}_{val}")

    # Vecteur final ordonné selon le modèle
    vec = np.zeros(len(model_features), dtype=np.float32)

    # Remplissage
    for i, name in enumerate(model_features):
        if name in NUMERIC_FIELDS:
            vec[i] = _to_float(client_doc.get(name, 0))
        elif name in one_hot_keys:
            vec[i] = 1.0
        else:
            # Cas particulier: certains numériques non listés mais présents 1:1
            # (sécurité si la liste NUMERIC_FIELDS est incomplète)
            if name in client_doc and isinstance(client_doc.get(name), (int, float, str)):
                vec[i] = _to_float(client_doc.get(name))
            else:
                vec[i] = 0.0

    return vec.reshape(1, -1)

# -----------------------------------------------------------------------------
# List clients
# -----------------------------------------------------------------------------
@router.get("/clients", response_model=ClientListResponse)
async def get_clients(
    skip: int = 0,
    limit: int = 20,
    gender: str | None = None,
    contract_type: str | None = None,
    income_type: str | None = None,
    education_type: str | None = None,
    family_status: str | None = None,
    housing_type: str | None = None,
    occupation_type: str | None = None,
    min_income: float | None = None,
    max_income: float | None = None,
    min_credit: float | None = None,
    max_credit: float | None = None,
    min_annuity: float | None = None,
    max_annuity: float | None = None,
    min_goods_price: float | None = None,
    max_goods_price: float | None = None,
    min_children: int | None = None,
    max_children: int | None = None,
    min_fam_members: float | None = None,
    max_fam_members: float | None = None,
    region_rating: int | None = None,
    target: int | None = None
):
    """
    Récupère une liste paginée de clients depuis MongoDB avec filtres.
    Réduit l'empreinte mémoire (projection champs utiles).
    """
    db = await get_database()
    collection_name = os.getenv("MONGO_COLL_CLIENTS", "client")

    try:
        # Construction du filtre
        query: Dict[str, Any] = {}
        if gender:
            query["CODE_GENDER"] = gender
        if contract_type:
            query["NAME_CONTRACT_TYPE"] = contract_type
        if income_type:
            query["NAME_INCOME_TYPE"] = income_type
        if education_type:
            query["NAME_EDUCATION_TYPE"] = education_type
        if family_status:
            query["NAME_FAMILY_STATUS"] = family_status
        if housing_type:
            query["NAME_HOUSING_TYPE"] = housing_type
        if occupation_type:
            query["OCCUPATION_TYPE"] = occupation_type
        if region_rating is not None:
            query["REGION_RATING_CLIENT"] = region_rating
        if target is not None:
            query["TARGET"] = target

        def add_range_filter(field: str, min_val: Any, max_val: Any):
            if min_val is not None or max_val is not None:
                sub = {}
                if min_val is not None:
                    sub["$gte"] = min_val
                if max_val is not None:
                    sub["$lte"] = max_val
                query[field] = sub

        add_range_filter("AMT_INCOME_TOTAL", min_income, max_income)
        add_range_filter("AMT_CREDIT",       min_credit, max_credit)
        add_range_filter("AMT_ANNUITY",      min_annuity, max_annuity)
        add_range_filter("AMT_GOODS_PRICE",  min_goods_price, max_goods_price)
        add_range_filter("CNT_CHILDREN",     min_children, max_children)
        add_range_filter("CNT_FAM_MEMBERS",  min_fam_members, max_fam_members)

        # Projection pour limiter la taille de réponse
        projection = {
            "_id": 0,
            "SK_ID_CURR": 1,
            "AMT_INCOME_TOTAL": 1,
            "CODE_GENDER": 1,
        }

        cursor = (
            db[collection_name]
            .find(query, projection=projection)
            .skip(max(0, skip))
            .limit(max(1, min(limit, 200)))  # garde une limite raisonnable
        )

        items: list[ClientItem] = []
        async for doc in cursor:
            raw_gender = doc.get("CODE_GENDER")
            gender_val = raw_gender if isinstance(raw_gender, str) else None
            items.append(
                ClientItem(
                    id=str(doc.get("SK_ID_CURR", "")),
                    revenue=doc.get("AMT_INCOME_TOTAL"),
                    gender=gender_val,
                )
            )

        # Compte total pour la pagination (sur le même filtre !)
        total_count = await db[collection_name].count_documents(query)
        pages = (total_count // limit) + (1 if total_count % limit else 0)
        has_next = (skip + limit) < total_count

        return ClientListResponse(
            items=items,
            page=(skip // limit) + 1,
            limit=limit,
            total=total_count,
            pages=pages,
            has_next=has_next,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {e}")

# -----------------------------------------------------------------------------
# Client details + prédiction
# -----------------------------------------------------------------------------
@router.get("/clients/{id}", response_model=ClientDetailResponse)
async def get_client_by_id(id: str):
    """
    Récupère les infos d'un client par son SK_ID_CURR + calcule la proba de défaut du modèle.
    Optimisé mémoire: pas de pandas; modèle chargé une fois; encodage one-hot manuel.
    """
    db = await get_database()
    collection_name = os.getenv("MONGO_COLL_CLIENTS", "client")

    try:
        try:
            sk_id = int(id)
        except ValueError:
            raise HTTPException(status_code=400, detail="SK_ID_CURR doit être un entier")

        client = await db[collection_name].find_one({"SK_ID_CURR": sk_id})
        if not client:
            raise HTTPException(status_code=404, detail="Client non trouvé")

        # Nettoyage NaN et retrait de _id
        client.pop("_id", None)
        client = _replace_nan(client)

        # Calcul score (proba défaut)
        score = 0.0
        try:
            model = load_model()
            model_features = getattr(model, "feature_names_in_", None)
            if model_features is None:
                raise RuntimeError("Le modèle ne contient pas 'feature_names_in_'.")
            X = to_feature_vector(client, model_features)
            proba = model.predict_proba(X)[0][1]
            score = float(proba)
        except Exception:
            score = 0.0  # fallback sécurisé

        # Mapping réponse
        detail = ClientDetail(
            id=str(client.get("SK_ID_CURR", "")),
            revenue=client.get("AMT_INCOME_TOTAL"),
            gender=client.get("CODE_GENDER"),
            raw=ClientRaw.model_validate(client),
        )

        thr = float(os.getenv("THRESHOLD", "0.5"))
        pred = 1 if score >= thr else 0
        decision = "REJECT" if pred == 1 else "ACCEPT"

        return ClientDetailResponse(
            client=detail,
            prediction=pred,            # classe 0/1
            proba_default=score,        # proba float
            threshold=thr,
            decision=decision,
            top_features=[],            # à remplir si tu ajoutes une explaina (SHAP, etc.)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {e}")
