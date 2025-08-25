
# Import des modules nécessaires
from fastapi import APIRouter, HTTPException
# Import de la fonction d'accès à la base MongoDB
from app.services.database import get_database
from app.schemas.clients  import ClientItem, ClientListResponse
from app.schemas.clients import  ClientDetailResponse, ClientDetail, ClientRaw
import os


# Création du router pour regrouper les routes liées aux clients
router = APIRouter()


@router.get("/clients", response_model=ClientListResponse)
async def get_clients(
    skip: int = 0,
    limit: int = 20,
    gender: str = None,
    contract_type: str = None,
    income_type: str = None,
    education_type: str = None,
    family_status: str = None,
    housing_type: str = None,
    occupation_type: str = None,
    min_income: float = None,
    max_income: float = None,
    min_credit: float = None,
    max_credit: float = None,
    min_annuity: float = None,
    max_annuity: float = None,
    min_goods_price: float = None,
    max_goods_price: float = None,
    min_children: int = None,
    max_children: int = None,
    min_fam_members: float = None,
    max_fam_members: float = None,
    region_rating: int = None,
    target: int = None
):
    """
    Récupère une liste de clients paginée depuis la base MongoDB.
    - skip : nombre de clients à ignorer (pour la pagination)
    - limit : nombre maximum de clients à retourner
    La réponse inclut aussi le nombre total, le nombre restant et les paramètres utilisés.
    """
    db = await get_database()  # Connexion à la base
    collection_name = os.getenv("MONGO_COLL_CLIENTS", "client")  # Nom de la collection
    try:
        # Construction dynamique du filtre
        query = {}
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
        if region_rating:
            query["REGION_RATING_CLIENT"] = region_rating
        if target is not None:
            query["TARGET"] = target

        # Filtres numériques (intervalle)
        def add_range_filter(field, min_val, max_val):
            if min_val is not None or max_val is not None:
                query[field] = {}
                if min_val is not None:
                    query[field]["$gte"] = min_val
                if max_val is not None:
                    query[field]["$lte"] = max_val

        add_range_filter("AMT_INCOME_TOTAL", min_income, max_income)
        add_range_filter("AMT_CREDIT", min_credit, max_credit)
        add_range_filter("AMT_ANNUITY", min_annuity, max_annuity)
        add_range_filter("AMT_GOODS_PRICE", min_goods_price, max_goods_price)
        add_range_filter("CNT_CHILDREN", min_children, max_children)
        add_range_filter("CNT_FAM_MEMBERS", min_fam_members, max_fam_members)

        # Récupération des clients avec pagination et filtre
        cursor = db[collection_name].find(query).skip(skip).limit(limit)
        clients = []

        # Fonction utilitaire pour remplacer les NaN par None (JSON compatible)
        import math
        def replace_nan(obj):
            if isinstance(obj, dict):
                return {k: replace_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan(v) for v in obj]
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            else:
                return obj

        # Parcours des résultats et conversion de l'_id + nettoyage des NaN
        async for client in cursor:
            # Contrôle du type de gender
            raw_gender = client.get("CODE_GENDER")
            if isinstance(raw_gender, str):
                gender = raw_gender
            else:
                gender = None
            item = ClientItem(
                id=str(client.get("SK_ID_CURR", "")),
                revenue=client.get("AMT_INCOME_TOTAL"),
                gender=gender
            )
            clients.append(item)

        # Calcul du nombre total et du nombre restant
        total_count = await db[collection_name].count_documents({})
        pages = (total_count // limit) + (1 if total_count % limit else 0)
        has_next = skip + limit < total_count
        return ClientListResponse(
            items=clients,
            page=(skip // limit) + 1,
            limit=limit,
            total=total_count,
            pages=pages,
            has_next=has_next
        )
    except Exception as e:
        # Gestion des erreurs : retourne un message d'erreur HTTP
        raise HTTPException(status_code=400, detail=f"Erreur: {e}")



# Route pour récupérer les infos d'un client par son SK_ID_CURR
@router.get("/clients/{id}", response_model=ClientDetailResponse)
async def get_client_by_id(id: str):
    """
    Récupère les infos détaillées d'un client par son identifiant SK_ID_CURR.
    """
    db = await get_database()
    collection_name = os.getenv("MONGO_COLL_CLIENTS", "client")
    try:
        # Conversion sécurisée de l'id en int
        try:
            sk_id = int(id)
        except ValueError:
            raise HTTPException(status_code=400, detail="SK_ID_CURR doit être un entier")

        client = await db[collection_name].find_one({"SK_ID_CURR": sk_id})
        if not client:
            raise HTTPException(status_code=404, detail="Client non trouvé")

        # Retire le champ '_id' pour éviter l'erreur Pydantic
        client.pop('_id', None)

        # Nettoie les NaN dans le document client
        import math
        def replace_nan(obj):
            if isinstance(obj, dict):
                return {k: replace_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan(v) for v in obj]
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            else:
                return obj
        client = replace_nan(client)

        # Calcul du score du modèle pkl
        import pickle
        import numpy as np
        import pandas as pd
        model_path = os.getenv("MODEL_PATH", "models_artifacts/model.pkl")
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
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
                "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
                "FLAG_DOCUMENT_COUNT"
            ]
            client_features = {k: client.get(k, 0) for k in features_list}
            df_client = pd.DataFrame([client_features])
            df_client = pd.get_dummies(df_client)
            model_features = getattr(model, 'feature_names_in_', None)
            if model_features is not None:
                for col in model_features:
                    if col not in df_client.columns:
                        df_client[col] = 0
                df_client = df_client[model_features]
            features = df_client.values
            score = float(model.predict_proba(features)[0][1])
        except Exception as e:
            score = 0.0  # Valeur par défaut en cas d'erreur

        # Mapping vers ClientDetail
        detail = ClientDetail(
            id=str(client.get("SK_ID_CURR", "")),
            revenue=client.get("AMT_INCOME_TOTAL"),
            gender=client.get("CODE_GENDER"),
            raw=ClientRaw.model_validate(client)
        )
        thr = float(os.getenv("THRESHOLD", "0.5"))
        pred = 1 if score >= thr else 0
        dec = "REJECT" if pred == 1 else "ACCEPT"
        return ClientDetailResponse(
            client=detail,
            prediction=int(score),      # conversion explicite
            proba_default=float(score), # conversion explicite
            threshold=thr,
            decision=dec,
            top_features=[]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {e}")
