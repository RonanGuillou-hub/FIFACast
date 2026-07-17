"""
Script d'entraînement réel

Contenu :
- FeatureEngineer (importée depuis feature_engineering.py, voir plus bas) :
  transformer sklearn custom, fait partie du pipeline car doit rester
  identique train/test/prod.
- ColumnTransformer : imputation + scaling + encodage, fit UNIQUEMENT
  sur le train (aucune fuite de données).
- Pipeline complet : feature engineering + preprocessing + modèle.
- Tracking MLflow (paramètres, métriques, modèle) + push du modèle
  entraîné vers le Hub HuggingFace.

Le chargement des données brutes et leur nettoyage structurel
(déduplication, correction des valeurs aberrantes évidentes, fusion 
nouvelles données) ont déjà eu lieu en amont, dans src/data/make_dataset.py.

Paramètres non-secrets (colonnes, hyperparamètres, noms de repos) lus
depuis configs/config.yaml — voir src/config.py. Les secrets (HF_TOKEN,
MLFLOW_TRACKING_URI) restent en variables d'environnement.
"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from huggingface_hub import HfApi
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config
from src.models.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG = load_config()

PROCESSED_DATA_PATH = (
    Path(CONFIG["paths"]["data"]["processed"]) / CONFIG["paths"]["processed_filename"]
)

RANDOM_STATE = CONFIG["training"]["random_state"]
TEST_SIZE = CONFIG["training"]["test_size"]
CV_FOLDS = CONFIG["training"]["cv_folds"]
SCORING = CONFIG["training"]["scoring"]

MODEL_PARAMS = CONFIG["model"]["params"]
NUMERIC_COLS = CONFIG["model"]["features"]["numeric"]
CATEGORICAL_COLS = CONFIG["model"]["features"]["categorical"]
TARGET_COL = CONFIG["model"]["target"]

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI") or CONFIG["mlflow"]["default_tracking_uri"]
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME") or CONFIG["mlflow"]["experiment_name"]

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO") or CONFIG["huggingface"]["model_repo"]
HF_TOKEN = os.environ.get("HF_TOKEN")

# ---------------------------------------------------------------------------
# 1. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def build_model() -> Pipeline:
    """Construit le pipeline complet : feature engineering + preprocessing + modèle."""

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])

    return Pipeline(steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(**MODEL_PARAMS)),
    ])


# ---------------------------------------------------------------------------
# 2. CHARGEMENT DES DONNÉES PRÉTRAITÉES
# ---------------------------------------------------------------------------
def load_data(path: Path = PROCESSED_DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lance d'abord `python -m src.data.make_dataset` "
            "pour générer les données prétraitées."
        )
    df = pd.read_csv(path)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# 3. ENTRAÎNEMENT + ÉVALUATION + VALIDATION CROISÉE
# ---------------------------------------------------------------------------
def train(model: Pipeline, X, y):
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # fit + transforme sur le train
    model.fit(X_train, y_train) 

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info("Accuracy (test) : %.3f", accuracy)
    logger.info("\n%s", classification_report(y_test, y_pred))

    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring=SCORING)
    logger.info("Accuracy (CV, %d folds) : %.3f (+/- %.3f)", CV_FOLDS, cv_scores.mean(), cv_scores.std())

    metrics = {
        "accuracy": accuracy,
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "precision_classe_1": report["1"]["precision"],
        "recall_classe_1": report["1"]["recall"],
        "f1_classe_1": report["1"]["f1-score"],
    }

    return model, metrics


# ---------------------------------------------------------------------------
# 4. MAIN : tracking MLflow + push du modèle vers le Hub HuggingFace
# ---------------------------------------------------------------------------
def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        try:
            X, y = load_data()
            model = build_model()

            params = {
                "model_type": CONFIG["model"]["type"],
                **MODEL_PARAMS,
                "test_size": TEST_SIZE,
                "cv_folds": CV_FOLDS,
            }
            mlflow.log_params(params)

            trained_model, metrics = train(model, X, y)
            mlflow.log_metrics(metrics)
          
            trusted_feature_engineer = f"{FeatureEngineer.__module__}.FeatureEngineer"

            model_info = mlflow.sklearn.log_model(
                trained_model,
                "model",
                code_paths=["src", "configs"],
                skops_trusted_types=[
                    trusted_feature_engineer,
                    "numpy.dtype",
                ],
                registered_model_name=CONFIG["mlflow"]["registered_model_name"],
            )

            # Publication du modèle sur MLFLOW
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(
                name=CONFIG["mlflow"]["registered_model_name"],
                alias=CONFIG["mlflow"]["model_alias"],
                version=model_info.registered_model_version,
            )
            logger.info(
                "Modèle enregistré : %s (version %s, alias @%s)",
                CONFIG["mlflow"]["registered_model_name"],
                model_info.registered_model_version,
                CONFIG["mlflow"]["model_alias"],
            )

        except Exception as e:
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            logger.exception("Échec de l'entraînement")
            raise

        # Sauvegarde locale
        import joblib

        model_filename = CONFIG["paths"]["model_filename"]
        fichier, extension = model_filename.rsplit(".", 1)  # ex: "model", "joblib"

        history_dir = Path(CONFIG["paths"]["models_history"])
        history_dir.mkdir(parents=True, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        history_path = history_dir / f"{fichier}_{timestamp}.{extension}"
        joblib.dump(trained_model, history_path)
        logger.info("Copie historisée sauvegardée dans %s", history_path)

        # --- Push du modèle vers le Hub HuggingFace (optionnel, si HF_TOKEN présent) ---
        if HF_TOKEN:
            api = HfApi(token=HF_TOKEN)

            model_dir = Path(CONFIG["paths"]["models"])
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / model_filename
            joblib.dump(trained_model, model_path)

            # Copie dernière version
            api.upload_file(
                path_or_fileobj=str(model_path),
                path_in_repo=model_filename,
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )

            # Copie historisée
            api.upload_file(
                path_or_fileobj=str(history_path),
                path_in_repo=f"history/{history_path.name}",
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )
            logger.info("Modèle poussé vers %s (+ copie historisée)", HF_MODEL_REPO)
        else:
            logger.warning("HF_TOKEN absent : le modèle n'a pas été poussé sur le Hub (reste disponible en local).")


if __name__ == "__main__":
    main()
