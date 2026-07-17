.PHONY: install data train predict test lint clean docker-build docker-train docker-notebook api streamlit

install:
	pip install -r requirements.txt

docker-build:
	docker compose build

docker-train:
	docker compose run --rm ml

docker-notebook:
	docker compose up notebook

data:
	python -m src.data.make_dataset

train:
	python -m src.models.train

predict:
	python -m src.models.predict_model $(if $(INPUT),--input $(INPUT))

test:
	pytest tests/

api:
	python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

streamlit:
	python -m streamlit run src/app/streamlit_app.py

lint:
	flake8 src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
