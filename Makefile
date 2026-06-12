.PHONY: install train serve test docker pipeline clean

install:
	pip install -e ".[dev]"

pipeline:
	python pipelines/run_pipeline.py --save-data

train:
	python -m credit_risk.train

serve:
	uvicorn credit_risk.api:app --reload --port 8000

test:
	pytest -v

docker:
	docker build -t credit-risk-mlops .

clean:
	rm -rf models/*.joblib models/*.json data/*.csv .pytest_cache
