"""
Schémas Pydantic — contrats d'entrée/sortie de l'API.

Observation reprend les colonnes BRUTES attendues par le pipeline
sklearn (model.features dans configs/config.yaml), à l'exception des
colonnes dérivées automatiquement par FeatureEngineer (revenu_par_age,
tranche_age, categorie_region...) et des colonnes météo (temperature_max/min,
precipitation), que l'API récupère elle-même via fetch_weather_for_region
en fonction de `region` et `date` — voir src/api/main.py.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    age: float = Field(..., ge=0, le=120, examples=[35])
    revenu: float = Field(..., ge=0, examples=[42000])
    anciennete_mois: int = Field(..., ge=0, examples=[18])
    categorie: str = Field(..., examples=["A"])
    region: str = Field(..., examples=["Nord"])
    date: Optional[str] = Field(
        None,
        description="Format AAAA-MM-JJ. Par défaut : aujourd'hui. Utilisée pour récupérer la météo de la région.",
        examples=["2026-07-15"],
    )


class PredictionResponse(BaseModel):
    prediction: int
    probabilite: float


class BatchObservations(BaseModel):
    observations: List[Observation]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    loaded: bool
    source: Optional[str] = None
    loaded_at: Optional[str] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
