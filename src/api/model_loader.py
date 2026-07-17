"""
Chargement du modèle servi par l'API, depuis l'une de 3 sources
(configs/config.yaml, section api.model_source) :
- "local"  : models/<model_filename>, pour le développement
- "s3"     : bucket S3 configuré (api.s3)
- "mlflow" : registre de modèles MLflow (api.mlflow)

Le modèle chargé est mis en cache en mémoire (process API), pour ne pas
le retélécharger à chaque requête. /model/reload force un rechargement.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)
CONFIG = load_config()

_cache: dict[str, Any] = {"model": None, "source": None, "loaded_at": None, "detail": None}


def _load_from_local() -> Any:
    model_path = Path(CONFIG["paths"]["models"]) / CONFIG["paths"]["model_filename"]
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} introuvable.")
    return joblib.load(model_path)


def _load_from_s3() -> Any:
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        raise RuntimeError("Credentials AWS absents (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).")

    import boto3

    s3_cfg = CONFIG["api"]["s3"]
    s3 = boto3.client("s3")

    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        s3.download_file(s3_cfg["bucket"], s3_cfg["key"], tmp.name)
        model = joblib.load(tmp.name)

    logger.info("Modèle chargé depuis s3://%s/%s", s3_cfg["bucket"], s3_cfg["key"])
    return model


def _load_from_mlflow() -> Any:
    import mlflow.sklearn

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or CONFIG["mlflow"]["default_tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    # URI par alias (models:/<nom>@<alias>), pas par stage -- les stages
    # MLflow (Staging/Production) sont dépréciés depuis la 2.9. L'alias
    # est assigné par train.py à chaque entraînement via
    # set_registered_model_alias(). Même nom/alias que côté enregistrement
    # (section mlflow: de config.yaml), pour ne jamais désynchroniser les deux.
    registered_name = CONFIG["mlflow"]["registered_model_name"]
    alias = CONFIG["mlflow"]["model_alias"]
    model_uri = f"models:/{registered_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Modèle chargé depuis MLflow : %s", model_uri)
    return model


_LOADERS = {
    "local": _load_from_local,
    "s3": _load_from_s3,
    "mlflow": _load_from_mlflow,
}


def load_model(source: Optional[str] = None, force_reload: bool = False) -> Any:
    """Charge (ou retourne depuis le cache) le modèle d'inférence."""
    source = source or CONFIG["api"]["model_source"]

    if _cache["model"] is not None and not force_reload and _cache["source"] == source:
        return _cache["model"]

    if source not in _LOADERS:
        raise ValueError(f"model_source inconnu : {source!r} (attendu 'local', 's3' ou 'mlflow')")

    logger.info("Chargement du modèle (source=%s)...", source)
    try:
        model = _LOADERS[source]()
    except Exception as e:
        _cache["detail"] = str(e)
        raise

    _cache["model"] = model
    _cache["source"] = source
    _cache["loaded_at"] = pd.Timestamp.now().isoformat()
    _cache["detail"] = None

    return model


def get_model_info() -> dict:
    return {
        "loaded": _cache["model"] is not None,
        "source": _cache["source"],
        "loaded_at": _cache["loaded_at"],
        "detail": _cache["detail"],
    }
