"""
API d'inférence — sert le modèle entraîné (local / S3 / MLflow selon
configs/config.yaml, section api.model_source).

Enrichit automatiquement chaque observation le fichier external des 
nouveaux match via fetch_result_by_new — même logique
qu'à l'entraînement, pour éviter toute divergence train/serving).

Lancement local :
    uvicorn src.api.main:app --reload

Documentation interactive une fois lancée : http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.model_loader import get_model_info, load_model
from src.api.schemas import (
    BatchObservations,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    Observation,
    PredictionResponse,
)
from src.data.make_dataset import fetch_weather_for_region

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Tente de précharger le modèle pour un premier appel rapide. Un échec
    ici n'empêche pas l'API de démarrer — /predict retentera le
    chargement à la volée (et remontera une erreur 503 explicite si la
    source est réellement indisponible).
    """
    try:
        load_model()

    except Exception as e:
        logger.warning("Préchargement du modèle échoué au démarrage : %s", e)

    yield


app = FastAPI(
    title="Foot CAST ! — API d'inférence",
    description="Sert les prédictions du modèle entraîné (source configurable : local, S3, MLflow).",
    version="1.0.0",
    lifespan=lifespan,
)

# Cache mémoire (région, date) -> météo, pour éviter de re-fetcher la même
# combinaison à chaque requête. Vidé au redémarrage du process — pas de
# persistance nécessaire, la météo d'un jour donné ne change pas.
_result_cache: Dict[Tuple[str, str], dict] = {}


def _get_result(region: str, date: str) -> dict:
    key = (region, date)
    if key in _result_cache:
        return _result_cache[key]

    try:
        result_df = fetch_weather_for_region(region, date, date)

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Échec de récupération météo pour région={region!r}, date={date!r} : {e}",
        )

    if result_df.empty:
        raise HTTPException(
            status_code=502,
            detail=f"Aucune donnée météo disponible pour région={region!r}, date={date!r}.",
        )

    row = result_df.iloc[0]

    weather = {
        "temperature_max": float(row["temperature_max"]),
        "temperature_min": float(row["temperature_min"]),
        "precipitation": float(row["precipitation"]),
    }


    _result_cache[key] = weather
    return weather


def _enrich_with_weather(observation: Observation) -> dict:
    date = observation.date or pd.Timestamp.now().strftime("%Y-%m-%d")
    weather = _get_weather(observation.region, date)
    return {**observation.model_dump(exclude={"date"}), **weather}


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health():
    """Vérifie que l'API répond — ne garantit pas que le modèle est chargé (voir /model/info)."""
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["monitoring"])
def model_info():
    """État du modèle actuellement en cache (source, date de chargement, dernière erreur éventuelle)."""
    return get_model_info()


@app.post("/model/reload", response_model=ModelInfoResponse, tags=["monitoring"])
def model_reload():
    """Force un rechargement du modèle depuis sa source configurée (utile après un nouvel entraînement)."""
    try:
        load_model(force_reload=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec du rechargement du modèle : {e}")
    return get_model_info()


def _predict(df: pd.DataFrame):
    try:
        model = load_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Modèle indisponible : {e}")

    try:
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur lors de la prédiction : {e}")

    return predictions, probabilities


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(observation: Observation):
    """Prédiction sur une observation unique. La météo de la région est récupérée automatiquement."""
    enriched = _enrich_with_weather(observation)
    df = pd.DataFrame([enriched])
    predictions, probabilities = _predict(df)
    return {"prediction": int(predictions[0]), "probabilite": float(probabilities[0])}


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
def predict_batch(batch: BatchObservations):
    """Prédiction sur plusieurs observations. La météo de chaque région/date est récupérée automatiquement (mise en cache par combinaison)."""
    if not batch.observations:
        raise HTTPException(status_code=400, detail="Aucune observation fournie.")

    enriched_rows = [_enrich_with_weather(obs) for obs in batch.observations]
    df = pd.DataFrame(enriched_rows)
    predictions, probabilities = _predict(df)

    results = [
        {"prediction": int(p), "probabilite": float(proba)}
        for p, proba in zip(predictions, probabilities)
    ]
    return {"predictions": results}
