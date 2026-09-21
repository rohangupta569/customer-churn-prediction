.PHONY: setup data train test app clean

setup:
	python3 -m pip install -e '.[dev,app]'

data:
	churn generate --rows 3000 --output data/customer_churn.csv

train:
	churn train --data data/customer_churn.csv --output-dir artifacts

test:
	pytest -q

app:
	streamlit run app.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -r {} +
	rm -rf .pytest_cache
