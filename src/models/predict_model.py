"""
Génération de prédictions sur des données fraîches, à partir du modèle
déjà entraîné (aucun entraînement n'a lieu ici).

Conçu pour tourner directement sur un runner GitHub Actions (CPU) :
contrairement à l'entraînement, l'inférence avec ce modèle ne nécessite
pas de GPU, donc pas besoin de passer par HuggingFace Jobs ici.

Flux :
    1. Téléchargement du modèle entraîné depuis le Hub HuggingFace (ou local)
    2. Chargement des données fraîches (colonnes brutes uniquement --
       PAS de colonnes météo, récupérées automatiquement à l'étape 3,
       exactement comme le fait l'API — voir src/api/main.py)
    3. Enrichissement météo (région + date -> temperature_max/min, precipitation)
    4. Prédiction (le pipeline sklearn complet - FeatureEngineer +
       ColumnTransformer inclus - gère tout automatiquement)
    5. Sauvegarde des résultats
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

from src.config import load_config
from src.data.make_dataset import fetch_weather_for_region

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG = load_config()

MODEL_DIR = Path(CONFIG["paths"]["models"])
MODEL_FILENAME = CONFIG["paths"]["model_filename"]
DEFAULT_OUTPUT = Path(CONFIG["paths"]["predictions_filename"])

# Secrets / valeurs dépendantes de l'environnement : variables d'env en
# priorité, config.yaml comme valeur par défaut non-secrète en repli.
# `or` (et non `.get(key, default)`) car une variable GitHub Actions
# non configurée arrive comme chaîne vide, pas absente.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO") or CONFIG["huggingface"]["model_repo"]
HF_TOKEN = os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# 1. CHARGEMENT DU MODÈLE (déjà entraîné, aucun apprentissage ici)
# ---------------------------------------------------------------------------
def load_model(source: str = "hub", model_path: Path = MODEL_DIR):
    """
    Charge le modèle entraîné.
    - source="hub"   : télécharge le modèle depuis HF_MODEL_REPO
    - source="local" : charge depuis models/<model_filename>
    """
    if source == "hub":
        logger.info("Téléchargement du modèle depuis %s", HF_MODEL_REPO)
        model_file = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=MODEL_FILENAME,
            token=HF_TOKEN,
        )
    elif source == "local":
        model_file = model_path / MODEL_FILENAME
        if not model_file.exists():
            raise FileNotFoundError(f"{model_file} introuvable.")
    else:
        raise ValueError(f"source inconnue : {source!r} (attendu 'hub' ou 'local')")

    return joblib.load(model_file)


# ---------------------------------------------------------------------------
# 2. CHARGEMENT DES DONNÉES FRAÎCHES
# ---------------------------------------------------------------------------
def load_input_data(input_path: Path = None) -> pd.DataFrame:
    """
    Charge les données fraîches à prédire.
    TODO: brancher la vraie source (API, base de données, fichier déposé
    périodiquement dans data/raw...). Pour l'instant, seul un fichier CSV
    passé explicitement via --input est supporté (mode test/manuel).

    Schéma attendu : colonnes brutes (age, revenu, anciennete_mois,
    categorie, region) + optionnellement 'date' (AAAA-MM-JJ, défaut :
    aujourd'hui). PAS de colonnes météo -- récupérées automatiquement
    par enrich_with_weather().
    """
    if input_path is not None:
        return pd.read_csv(input_path)

    raise NotImplementedError(
        "Aucune source de données fraîches configurée — branche ici l'appel "
        "à ton API / base de données / fichier déposé périodiquement."
    )


# ---------------------------------------------------------------------------
# 3. ENRICHISSEMENT MÉTÉO (même logique que l'API, voir src/api/main.py)
# ---------------------------------------------------------------------------
# Cache mémoire (région, date) -> météo, pour éviter de re-fetcher la même
# combinaison plusieurs fois dans un même batch.
_weather_cache: Dict[Tuple[str, str], dict] = {}


def _get_weather(region: str, date: str) -> dict:
    key = (region, date)
    if key in _weather_cache:
        return _weather_cache[key]

    weather_df = fetch_weather_for_region(region, date, date)
    if weather_df.empty:
        raise RuntimeError(f"Aucune donnée météo disponible pour région={region!r}, date={date!r}.")

    row = weather_df.iloc[0]
    weather = {
        "temperature_max": float(row["temperature_max"]),
        "temperature_min": float(row["temperature_min"]),
        "precipitation": float(row["precipitation"]),
    }
    _weather_cache[key] = weather
    return weather


def enrich_with_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute temperature_max/min et precipitation à chaque ligne, en
    fonction de sa région et de sa date ('date' absente ou vide ->
    aujourd'hui). Un seul appel API par combinaison (région, date)
    distincte présente dans le batch, grâce au cache.
    """
    df = df.copy()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    if "date" not in df.columns:
        df["date"] = today
    else:
        df["date"] = df["date"].fillna(today).replace("", today)

    weather_cols = df.apply(
        lambda row: pd.Series(_get_weather(row["region"], str(row["date"]))),
        axis=1,
    )
    return pd.concat([df, weather_cols], axis=1)


# ---------------------------------------------------------------------------
# 4. PRÉDICTION
# ---------------------------------------------------------------------------
def predict(model, data: pd.DataFrame) -> pd.DataFrame:
    """Le pipeline sklearn complet gère feature engineering + preprocessing + prédiction."""
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    result = data.copy()
    result["prediction"] = predictions
    result["probabilite"] = probabilities
    return result


def save_predictions(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Génère des prédictions sur des données fraîches.")
    parser.add_argument("--source", choices=["local", "hub"], default="hub", help="Origine du modèle")
    parser.add_argument("--input", type=Path, default=None, help="Fichier de données fraîches à prédire")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Fichier de sortie")
    args = parser.parse_args()

    model = load_model(source=args.source)
    data = load_input_data(args.input)
    data = enrich_with_weather(data)
    results = predict(model, data)

    output_file = save_predictions(results, args.output)
    logger.info("Prédictions sauvegardées dans %s (%d lignes)", output_file, len(results))


if __name__ == "__main__":
    main()
