"""
Chargement et nettoyage structurel des données.

Transforme les données brutes (data/raw) en données prêtes à l'emploi
pour le feature engineering / l'entraînement (data/processed).

Le nettoyage effectué ici est STRUCTUREL uniquement : il ne dépend
d'aucune statistique apprise (pas de moyenne, médiane, encodage...) et
ne crée donc aucune fuite de données entre train et test. C'est pour
cette raison qu'il vit ici, hors du pipeline sklearn, et s'applique
une seule fois sur l'ensemble du dataset avant le split train/test.

Les transformations qui, elles, dépendent de statistiques apprises sur
le train (imputation, scaling, encodage) vivent dans src/models/train.py,
à l'intérieur du pipeline sklearn.

Tous les paramètres non-secrets (chemins, régions, dataset synthétique)
viennent de configs/config.yaml — voir src/config.py.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG = load_config()


DATA_URL = CONFIG["external"]["match_result"]["csv_url"]

PROCESSED_DATA_DIR = Path(CONFIG["paths"]["data"]["processed"])
PROCESSED_FILENAME = CONFIG["paths"]["processed_filename"]
RAW_DATA_DIR = Path(CONFIG["paths"]["data"]["raw"])
RAW_FILENAME = CONFIG["paths"]["raw_filename"]


REQUIRED_COLUMNS = ["date","home_team","away_team","home_score","away_score","tournament","city","country","neutral"]

# ---------------------------------------------------------------------------
# 1. RECUPERATION DU FIHCIER HISTORIQUE
# ---------------------------------------------------------------------------
def load_raw_data():
    """Load the international results dataset and parse dates.

    Arguments:
    path -- location of the results.csv file

    Returns:
    df -- DataFrame with one row per international match, sorted by date,
          with an extra 'year' column
    """
    try:
        df = pd.read_csv(DATA_URL)
        df.to_csv(Path(RAW_DATA_DIR, RAW_FILENAME), index=False, sep=";", encoding="utf-8-sig")

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Le fichier '{DATA_URL}' est introuvable."
        ) from None

    return df

# ---------------------------------------------------------------------------
# 2. NETTOYAGE STRUCTUREL (avant le split, ne dépend d'aucune statistique apprise)
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrections qui ne "voient" pas la cible et ne créent pas de fuite de
    données :
    - validation du schéma attendu
    - suppression des doublons
    - correction des valeurs aberrantes évidentes (bornes physiques connues)
    - typage correct des colonnes
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset brut : {missing}")

    df = df.copy()
  
    # nombre de but par match
    n_before = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    logger.info("%d NaN supprimés", n_before - len(df)) 
    ["total_goals"] = (df["home_score"] + df["away_score"]).astype(int)

    # suppression des outliers
    high_band   = df["total_goals"].mean() + df["total_goals"].std()*3
    low_band    = df["total_goals"].mean() - df["total_goals"].std()*3
    total_goals_index = df[~((df["total_goals"] > low_band) & (df["total_goals"] < high_band))].index
    n_before = len(df)
    df.drop(total_goals_index, inplace=True)
    logger.info("%d Outliers supprimés", n_before - len(df)) 

    # année
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    
    # doublons complets
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("%d doublons supprimés", n_before - len(df))
    
    # Que les matches récents
    recent_match = (df["date"]>="01/01/1994") & (df["date"]<="30/06/2026")
    old_index = df[~recent_match].index
    n_before = len(df)
    df.drop(old_index, inplace=True)
    logger.info("%d Vieux matchs supprimés", n_before - len(df)) 

    # avantage sur le terrain a domicile
    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] < df["away_score"]
    ]
    choices = ["home_win" ,"away_win"]
    df["outcome"] = np.select(conditions, choices, default="draw") # assigne draw si les conditions ne sont pas respectées

    # equipe qui gagne
    df["team_win"] = np.where (df["outcome"]=="home_win", 
                                df["home_team"],  
                                np.where(df["outcome"] == "away_win",
                                    df["away_team"], 
                                    np.nan))
        
    # tri
    df = (
        df
        .reset_index(drop=True)
        .sort_values("date").reset_index(drop=True)
    )

    return df


def save_processed_data(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Sauvegarde le dataset nettoyé au format CSV, puis le persiste sur S3
    (paths.s3) — le disque de l'instance GPU HuggingFace est éphémère et
    disparaît à la fin du job, sans cette persistance il serait impossible
    d'auditer/rejouer un entraînement passé.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / PROCESSED_FILENAME
    df.to_csv(output_file, index=False)

    s3_cfg = CONFIG.get("s3")
    if s3_cfg:
        upload_file_to_s3(output_file, s3_cfg["bucket"], s3_cfg["prefixes"]["processed"])

    return output_file


# ---------------------------------------------------------------------------
# 3. DONNÉES EXTERNES : MÉTÉO (appel API à chaque exécution)
# ---------------------------------------------------------------------------
def fetch_result_by_new(region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Appelle l'API météo configurée (Open-Meteo par défaut, gratuite, sans
    clé) pour récupérer la météo journalière d'une région sur une plage
    de dates donnée.

    Tous les paramètres (URL, variables, timezone, régions/coordonnées)
    viennent de configs/config.yaml (section external.weather). Si un
    futur fournisseur nécessite une clé API, la passer via une variable
    d'environnement, jamais en dur dans le code ou dans le YAML.
    """
    import requests  # import local pour ne pas alourdir les autres scripts qui n'en ont pas besoin

    weather_cfg = CONFIG["external"]["weather"]
    regions = weather_cfg["regions"]

    coords = regions.get(region)
    if coords is None:
        raise ValueError(f"Région inconnue : {region!r} — ajoute ses coordonnées dans configs/config.yaml")

    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(weather_cfg["daily_variables"]),
        "timezone": weather_cfg["timezone"],
    }

    response = requests.get(weather_cfg["api_url"], params=params, timeout=30)
    response.raise_for_status()
    daily = response.json()["daily"]

    return pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "region": region,
        "temperature_max": daily["temperature_2m_max"],
        "temperature_min": daily["temperature_2m_min"],
        "precipitation": daily["precipitation_sum"],
    })


def upload_file_to_s3(local_file: Path, bucket: str, key_prefix: str) -> None:
    """
    Persiste un fichier local dans le bucket S3 configuré, sous la clé :
        {key_prefix}/{année}/{mois}/{nom}_{AAAAMMJJ}.csv
    où {nom} est déduit du dernier segment de key_prefix (ex: "meteo" pour
    key_prefix="external/meteo").

    Nécessite AWS_ACCESS_KEY_ID et AWS_SECRET_ACCESS_KEY en variables
    d'environnement — jamais dans config.yaml. Si ces credentials sont
    absents (ex: développement local), l'upload est simplement ignoré
    avec un warning.

    Note : la clé ne contient que la date (pas l'heure) — plusieurs runs
    le même jour écraseront le même fichier S3. Si tu as besoin d'un
    fichier distinct par run, ajoute l'heure dans le format ci-dessous.
    """
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        logger.warning("Credentials AWS absents — upload S3 de %s ignoré.", local_file.name)
        return

    import boto3  # import local pour ne pas alourdir les autres scripts qui n'en ont pas besoin

    now = pd.Timestamp.now()
    name = Path(key_prefix).name
    s3_key = f"{key_prefix}/{now:%Y}/{now:%m}/{name}_{now:%Y%m%d}.csv"

    s3 = boto3.client("s3")
    s3.upload_file(str(local_file), bucket, s3_key)
    logger.info("Fichier persisté sur s3://%s/%s", bucket, s3_key)


def fetch_external_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Récupère la météo pour toutes les combinaisons (région, plage de dates)
    présentes dans le dataset principal. Appelée à chaque exécution de
    make_dataset.py, donc toujours à jour — pas de dépendance à un cache
    pour fonctionner.

    Le résultat est sauvegardé localement (external.weather.cache_file)
    puis persisté sur S3 (external.weather.s3), car le disque de
    l'instance GPU HuggingFace est éphémère et disparaît à la fin du job.
    """
    start_date = df["date"].min().strftime("%Y-%m-%d")
    end_date = df["date"].max().strftime("%Y-%m-%d")

    frames = []
    for region in df["region"].dropna().unique():
        logger.info("Récupération météo pour %s (%s → %s)", region, start_date, end_date)
        try:
            frames.append(fetch_weather_for_region(region, start_date, end_date))
        except Exception as e:
            logger.warning("Échec de récupération météo pour %s : %s", region, e)

    if not frames:
        raise RuntimeError("Aucune donnée météo récupérée — vérifie la connectivité à l'API météo.")

    external_df = pd.concat(frames, ignore_index=True)

    cache_file = Path(CONFIG["external"]["weather"]["cache_file"])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    external_df.to_csv(cache_file, index=False)

    s3_cfg = CONFIG.get("s3")
    if s3_cfg:
        upload_file_to_s3(cache_file, s3_cfg["bucket"], s3_cfg["prefixes"]["weather"])

    return external_df


def merge_external_data(df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne le dataset principal avec les données météo, sur (date, région).

    Une jointure comme celle-ci reste hors du pipeline sklearn : elle est
    déterministe, ne dépend d'aucune statistique apprise sur le train, et
    ne crée donc pas de fuite de données.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    external_df = external_df.copy()
    external_df["date"] = pd.to_datetime(external_df["date"])

    merged = df.merge(external_df, on=["date", "region"], how="left")

    missing = merged["temperature_max"].isna().sum()
    if missing > 0:
        logger.warning("%d lignes sans correspondance météo après fusion", missing)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Prépare le dataset pour l'entraînement.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR, help="Dossier des données brutes")
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR, help="Dossier de sortie")
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Ne pas appeler l'API météo (utile pour un test rapide hors ligne)",
    )
    args = parser.parse_args()

    df = load_raw_data(args.input)
    df = clean_data(df)

    if not args.skip_external:
        external_df = fetch_external_data(df)
        df = merge_external_data(df, external_df)
    else:
        logger.info("Fusion météo ignorée (--skip-external)")

    output_file = save_processed_data(df, args.output)

    logger.info("Données traitées sauvegardées dans %s (%d lignes)", output_file, len(df))


if __name__ == "__main__":
    main()
