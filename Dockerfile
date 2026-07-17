# --- Image de base ---
FROM python:3.11-slim

# --- Variables d'environnement ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# --- Répertoire de travail ---
WORKDIR /app

# --- Dépendances système minimales (build tools, etc.) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Installation des dépendances Python (mise en cache du layer) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copie du code source ---
COPY . .

# --- Commande par défaut ---
CMD ["python", "-m", "src.models.train"]
