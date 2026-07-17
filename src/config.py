"""
Chargement centralisé de la configuration du projet.

Deux sources, deux natures différentes :
- configs/config.yaml : paramètres NON-secrets (hyperparamètres, chemins,
  noms de ressources), versionnés dans git.
- .env (racine du projet, jamais commité — voir .gitignore) : secrets
  (HF_TOKEN, MLFLOW_TRACKING_URI, AWS_ACCESS_KEY_ID...), chargés dans
  os.environ. En CI/production, ces mêmes variables sont injectées
  directement par la plateforme (GitHub Secrets, HuggingFace Jobs) —
  .env ne sert qu'au développement local.

Les scripts continuent de lire les secrets via os.environ.get(...), pas
via load_config() — ce module se contente de garantir que .env a été
chargé dans os.environ avant que ces appels aient lieu.
"""

from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

# override=False : si une variable existe déjà dans l'environnement réel
# (GitHub Actions, Docker, HuggingFace Jobs...), elle n'est jamais écrasée
# par .env. Silencieux si .env est absent (cas normal en CI/production).
load_dotenv(dotenv_path=ENV_PATH, override=False)


@lru_cache
def load_config(path: Path = CONFIG_PATH) -> dict:
    """Charge et met en cache le fichier de configuration YAML (paramètres non-secrets)."""
    with open(path) as f:
        return yaml.safe_load(f)
