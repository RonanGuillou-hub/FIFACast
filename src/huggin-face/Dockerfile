# Build a python 3.10 image
FROM python:3.10

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

#Les variables sont a ajouter dans les variables sur HF
CMD mlflow server -p $PORT \
--host 0.0.0.0 \
--allowed-hosts ronan-guillou-mlflow-test.hf.space \ 
--cors-allowed-origins "https://ronan-guillou-mlflow-test.hf.space" \
# base de données
--backend-store-uri $BACKEND_STORE_URI \
# storage S3
--default-artifact-root $ARTIFACT_STORE_URI