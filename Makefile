# Variables
SCRIPTS_DIR = scripts
SRC_DIR = src
DATA_DIR = data

# EXPORT THE PROJECT ROOT TO PYTHON'S PATH
export PYTHONPATH = .

.PHONY: data merge validate_raw validate_preprocessed preprocess pipeline test format lint check

data:
	@echo "Fetching Kaggle dataset..."
	poetry run python $(SCRIPTS_DIR)/data/download_kaggle.py
	@echo "Running AutoScout24 scraper..."
	poetry run bash scripts/data/scrape_data.sh
	@echo "Data acquisition complete!"

merge:
	@echo "Merging datasets..."
	poetry run python scripts/merge_data.py
	@echo "Running preprocessing pipeline..."
	poetry run python scripts/run_preprocessing.py

validate_raw:
	@echo "Validating data..."
	poetry run python scripts/data/validate_data.py raw

validate_preprocessed:
	@echo "Validating preprocessed data..."
	poetry run python scripts/data/validate_data.py preprocessed

clean:
	@echo "Cleaning data..."
	poetry run python scripts/data/clean_data.py

delete:
	@echo "Cleaning up intermediate files..."
	rm -rf $(DATA_DIR)/iterim
	rm -rf __pycache__ .pytest_cache dist build *.egg-info
	@echo "Cleanup complete!"

format:
	@echo "Formatting code with Black..."
	poetry run black $(SRC_DIR) $(SCRIPTS_DIR)
	@echo "Code formatting complete!"

lint:
	@echo "Linting code with Flake8..."
	poetry run flake8 $(SRC_DIR) $(SCRIPTS_DIR)
	@echo "Code linting complete!"

test:
	@echo "Running unit and integration tests with coverage..."
	poetry run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
	@echo "Detailed coverage report generated in htmlcov/index.html"

check: format lint
	@echo "Code quality checks passed!"

pipeline: merge validate_raw clean check delete
	@echo "Full data pipeline executed successfully!"