from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict

from app.schemas.base import StrictBase


# -------------------------------------------------------------------
# 1) Colonnes BRUTES conservées depuis application_test.csv
# -------------------------------------------------------------------
class ClientRaw(BaseModel):
    """
    Colonnes brutes (subset) issues de application_test.csv.
    Types choisis pour accepter valeurs int/float/None selon les cas.
    """
    model_config = ConfigDict(extra="forbid")

    SK_ID_CURR: int | None = None
    TARGET: int | None = None
    NAME_CONTRACT_TYPE: str | None = None
    CODE_GENDER: str | None = None
    FLAG_OWN_CAR: str | None = None
    FLAG_OWN_REALTY: str | None = None
    CNT_CHILDREN: int | None = None
    AMT_INCOME_TOTAL: float | None = None
    AMT_CREDIT: float | None = None
    AMT_ANNUITY: float | None = None
    AMT_GOODS_PRICE: float | None = None
    NAME_INCOME_TYPE: str | None = None
    NAME_EDUCATION_TYPE: str | None = None
    NAME_FAMILY_STATUS: str | None = None
    NAME_HOUSING_TYPE: str | None = None
    REGION_POPULATION_RELATIVE: float | None = None
    DAYS_BIRTH: int | None = None
    DAYS_EMPLOYED: int | None = None
    OWN_CAR_AGE: float | None = None
    FLAG_MOBIL: int | None = None
    FLAG_EMP_PHONE: int | None = None
    FLAG_CONT_MOBILE: int | None = None
    FLAG_EMAIL: int | None = None
    OCCUPATION_TYPE: str | None = None
    CNT_FAM_MEMBERS: float | None = None
    REGION_RATING_CLIENT: int | None = None
    REGION_RATING_CLIENT_W_CITY: int | None = None
    EXT_SOURCE_1: float | None = None
    EXT_SOURCE_2: float | None = None
    EXT_SOURCE_3: float | None = None
    HOUSETYPE_MODE: str | None = None
    TOTALAREA_MODE: float | None = None
    DEF_30_CNT_SOCIAL_CIRCLE: float | None = None
    DEF_60_CNT_SOCIAL_CIRCLE: float | None = None
    DAYS_LAST_PHONE_CHANGE: float | None = None
    AMT_REQ_CREDIT_BUREAU_HOUR: int | None = None
    AMT_REQ_CREDIT_BUREAU_DAY: int | None = None
    AMT_REQ_CREDIT_BUREAU_WEEK: int | None = None
    AMT_REQ_CREDIT_BUREAU_MON: int | None = None
    AMT_REQ_CREDIT_BUREAU_QRT: int | None = None
    AMT_REQ_CREDIT_BUREAU_YEAR: int | None = None
    FLAG_DOCUMENT_COUNT: int | None = None


# -------------------------------------------------------------------
# 2) Document normalisé pour MongoDB / API
#    - id: string (depuis SK_ID_CURR)
#    - revenue: float | None (depuis AMT_INCOME_TOTAL)
#    - gender: "M"/"F" | None (depuis CODE_GENDER, normalisé)
#    - raw: ClientRaw (toutes les colonnes conservées ci-dessus)
# -------------------------------------------------------------------
class ClientDoc(StrictBase):
    id: str
    revenue: float | None = Field(default=None, description="Mapped from AMT_INCOME_TOTAL")
    gender: str | None = Field(default=None, description="Normalized from CODE_GENDER to 'M'/'F'")
    raw: ClientRaw


# -------------------------------------------------------------------
# 3) Schémas d’API (liste / détail) – inchangés côté contrat
# -------------------------------------------------------------------
class ClientItem(StrictBase):
    id: str
    revenue: float | None = None
    gender: str | None = Field(default=None, description="M or F")


class ClientListResponse(StrictBase):
    items: list[ClientItem]
    page: int
    limit: int
    total: int
    pages: int
    has_next: bool


class ClientDetail(StrictBase):
    id: str
    revenue: float | None = None
    gender: str | None = None
    raw: ClientRaw


class ClientDetailResponse(StrictBase):
    client: ClientDetail
    prediction: int
    proba_default: float
    threshold: float
    decision: str
    top_features: list


# -------------------------------------------------------------------
# 4) Helper de mapping (optionnel)
#    Convertit un dict brut (ou ClientRaw) -> ClientDoc normalisé
#    - id: str(SK_ID_CURR)
#    - revenue: AMT_INCOME_TOTAL
#    - gender: CODE_GENDER normalisé en 'M'/'F' (autres valeurs => None)
# -------------------------------------------------------------------
def _normalize_gender(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip().upper()
    if s in {"F", "FEMALE"}:
        return "F"
    if s in {"M", "MALE"}:
        return "M"
    # Exemples: "XNA" -> None
    return None


def raw_to_client_doc(raw: dict | ClientRaw) -> ClientDoc:
    if isinstance(raw, dict):
        raw_obj = ClientRaw.model_validate(raw)
    else:
        raw_obj = raw

    return ClientDoc(
        id=str(raw_obj.SK_ID_CURR) if raw_obj.SK_ID_CURR is not None else "",
        revenue=float(raw_obj.AMT_INCOME_TOTAL) if raw_obj.AMT_INCOME_TOTAL is not None else None,
        gender=_normalize_gender(raw_obj.CODE_GENDER),
        raw=raw_obj,
    )
