"""
Interface Streamlit de base — consomme l'API FastAPI (src/api/main.py),
n'appelle jamais le modèle directement.

Lancement local :
    streamlit run src/app/streamlit_app.py
"""

import os

import requests
import streamlit as st

from src.config import load_config

CONFIG = load_config()

# En Docker Compose, l'API n'est pas sur localhost — API_URL (variable
# d'environnement) prend le dessus sur la valeur par défaut de config.yaml.
API_URL = os.environ.get("API_URL") or CONFIG["streamlit"]["api_url"]

st.set_page_config(page_title="Mon Projet ML", page_icon="🔮")
st.title("🔮 Prédiction — Mon Projet ML")
st.caption(f"API : {API_URL}")

with st.sidebar:
    st.header("Statut")

    if st.button("Vérifier la connexion à l'API"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            r.raise_for_status()
            st.success("API disponible ✅")
        except Exception as e:
            st.error(f"API indisponible : {e}")

    if st.button("Infos sur le modèle chargé"):
        try:
            r = requests.get(f"{API_URL}/model/info", timeout=5)
            r.raise_for_status()
            st.json(r.json())
        except Exception as e:
            st.error(f"Erreur : {e}")

    if st.button("Recharger le modèle"):
        try:
            r = requests.post(f"{API_URL}/model/reload", timeout=30)
            r.raise_for_status()
            st.success("Modèle rechargé")
            st.json(r.json())
        except requests.HTTPError as e:
            st.error(f"Erreur API ({e.response.status_code}) : {e.response.json().get('detail', e)}")
        except Exception as e:
            st.error(f"Erreur : {e}")

st.subheader("Nouvelle observation")
st.caption("La météo de la région/date est récupérée automatiquement par l'API.")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Âge", min_value=0, max_value=120, value=35)
    revenu = st.number_input("Revenu", min_value=0, value=42000, step=1000)
    anciennete_mois = st.number_input("Ancienneté (mois)", min_value=0, value=18)
with col2:
    categorie = st.selectbox("Catégorie", ["A", "B", "C"])
    region = st.selectbox("Région", ["Nord", "Sud", "Est", "Ouest"])
    date = st.date_input("Date (pour la météo)")

if st.button("Prédire", type="primary"):
    payload = {
        "age": age,
        "revenu": revenu,
        "anciennete_mois": anciennete_mois,
        "categorie": categorie,
        "region": region,
        "date": date.isoformat(),
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()

        col_a, col_b = st.columns(2)
        col_a.metric("Prédiction", result["prediction"])
        col_b.metric("Probabilité (classe 1)", f"{result['probabilite']:.1%}")
    except requests.HTTPError as e:
        st.error(f"Erreur API ({e.response.status_code}) : {e.response.json().get('detail', e)}")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
