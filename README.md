# FIFA Cast

Qui gagnera la coupe du monde foott 2026 ?

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

## Structure du projet

```
mon-projet-ml/
├── data/            # Données (raw, interim, processed, external)
├── notebooks/       # Notebooks d'exploration
├── src/             # Code source (data, features, models, evaluation, visualization)
├── models/          # Modèles entraînés
├── reports/         # Rapports et figures générés
├── configs/         # Fichiers de configuration
└── tests/           # Tests unitaires
```

## Usage

```bash
python -m src.data.make_dataset
python -m src.models.train
python -m src.models.predict_model
```

## Tests

```bash
pytest tests/
```

La suite couvre : `FeatureEngineer` (feature engineering), `make_dataset.py`
(nettoyage structurel, fusion météo), `train.py` (construction du pipeline,
entraînement, métriques), `predict_model.py` (chargement du modèle,
inférence), l'API FastAPI (tous les endpoints, météo mockée), et
`config.py`. Fixtures partagées dans `tests/conftest.py` — le modèle
d'entraînement (coûteux) n'est construit qu'une seule fois par session
de test (`trained_model`, scope `session`).

```bash
pytest tests/                    # tests unitaires uniquement (rapide, hors ligne)
pytest tests/ -m integration      # tests d'intégration uniquement (réseau requis)
pytest tests/ -m ""                # tous les tests
```

## Notebook — exécution manuelle du pipeline

`notebooks/manual_pipeline_run.ipynb` déroule tout le pipeline étape par
étape (génération/nettoyage des données, météo, feature engineering,
entraînement, inférence, test de l'API), utile pour déboguer ou explorer
sans passer par le cron GitHub Actions ni HuggingFace Jobs. Si l'API météo
n'est pas accessible (pas de réseau), le notebook retombe automatiquement
sur des valeurs factices pour pouvoir continuer hors ligne.

## Configuration (configs/config.yaml)

Tous les paramètres **non-secrets** du projet sont centralisés dans
`configs/config.yaml` : chemins, colonnes du modèle, hyperparamètres,
paramètres d'entraînement, régions/coordonnées météo, noms de repos
HuggingFace, nom d'expérience MLflow.

Chargement via `src/config.py` :

```python
from src.config import load_config
config = load_config()
```

**Ce qui reste en variable d'environnement (jamais dans ce fichier) :**

- `HF_TOKEN` : token HuggingFace
- `MLFLOW_TRACKING_URI` : URL du serveur MLflow (une valeur par défaut
  non-secrète existe dans `config.yaml`, mais la variable d'environnement
  prend toujours le dessus si elle est définie)
- `HF_MODEL_REPO` : peut aussi être surchargé via variable d'environnement
  si tu veux pointer temporairement vers un autre repo sans toucher au YAML

Pour changer un hyperparamètre (ex: `test_size`, `max_iter`) ou ajouter une
région météo, il suffit d'éditer `configs/config.yaml` — aucun code à
toucher.

### Modifier MLFLOW_TRACKING_URI (ou HF_MODEL_REPO, HF_TOKEN)

Ces valeurs vivent à 3 endroits différents, à mettre à jour séparément
selon où le code s'exécute :

1. **En local** : un fichier `.env` à la racine du projet (copier
   `.env.example`, jamais commité — voir `.gitignore`), chargé
   automatiquement par `src/config.py` (`python-dotenv`) dans
   `os.environ`. Une vraie variable d'environnement déjà définie n'est
   jamais écrasée par `.env` (`override=False`).

## Docker

Construire l'image et lancer l'entraînement dans un conteneur :

```bash
docker compose build
docker compose run --rm ml
```
Lancer un Jupyter Notebook accessible sur `http://localhost:8888` :

```bash
docker compose up notebook
```

## Entraînement automatisé (GPU à la demande + MLflow + Hub)

**Historisation des modèles entraînés** : chaque run sauvegarde deux
copies :
- `models/<model_filename>` (ex: `model.joblib`) — la version "courante",
  **écrasée à chaque run**, c'est celle que `predict_model.py` charge par
  défaut en `source="local"`
- `models/history/<model_filename_sans_extension>_<AAAAMMJJ_HHMMSS>.joblib`
  — une copie **datée, jamais écrasée**, pour retrouver n'importe quelle
  version entraînée précédemment

Si `HF_TOKEN` est présent, la copie historisée est aussi poussée sur le
Hub sous `history/<nom_du_fichier_daté>`, en plus de la version courante.

⚠️ Ni `models/` ni `models/history/` ne sont persistés au-delà d'un run sur
l'instance GPU HuggingFace (éphémère) — seul le push vers le Hub (ou S3, si
tu appliques le même mécanisme que pour `dataset_clean.csv`/`meteo.csv`)
survit à la fin du job.

**Model Registry MLflow (obligatoire pour que l'API charge le modèle)** :
`train.py` enregistre le modèle sous `mlflow.registered_model_name`
(`config.yaml`) et lui assigne l'alias `mlflow.model_alias` (ex:
`@champion`) à chaque run — sans ça, l'API (`api.model_source: "mlflow"`)
échoue avec `RESOURCE_DOES_NOT_EXIST: Registered Model ... not found`.
Utilise des **alias**, pas les stages `Staging`/`Production` (dépréciés
par MLflow depuis la version 2.9).

**Variables d'environnement à définir côté HuggingFace Jobs** (dans `trigger_job.py`
ou via la configuration du job) :

- `MLFLOW_TRACKING_URI` : URL de ton serveur MLflow
- `MLFLOW_EXPERIMENT_NAME` : nom de l'expérience MLflow
- `HF_MODEL_REPO` : repo HuggingFace où pousser le modèle entraîné
- `HF_TOKEN` : même token que ci-dessus, transmis au job

### Persistance sur S3

```
{prefix}/{année}/{mois}/{nom}_{AAAAMMJJ}.csv
```

## API d'inférence (FastAPI) + interface Streamlit

`src/api/` sert le modèle entraîné via une API REST. Contrairement à
`predict_model.py` (batch, déclenché par cron), l'API répond en temps réel
à des requêtes individuelles.

**Source du modèle** (`configs/config.yaml`, `api.model_source`) : `local`,
`s3` ou `mlflow` — interchangeable sans toucher au code.

### Lancer en local

```bash
make api          # démarre l'API sur http://localhost:8000 (docs interactives : /docs)
make streamlit     # démarre l'interface sur http://localhost:8501, dans un autre terminal
```

### Lancer avec Docker Compose

```bash
docker compose up api streamlit
```

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | Vérifie que l'API répond |
| GET | `/model/info` | Source et date de chargement du modèle en cache |
| POST | `/model/reload` | Force un rechargement depuis la source configurée |
| POST | `/predict` | Prédiction sur une observation unique |
| POST | `/predict/batch` | Prédiction sur plusieurs observations |

## Prédictions automatisées (données fraîches, sans réentraînement)

**Tester manuellement** avec l'exemple fourni :

```bash
make predict INPUT=data/examples/sample_input.csv
# ou directement :
python -m src.models.predict_model --source local --input data/examples/sample_input.csv
```