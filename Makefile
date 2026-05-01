# Variables
KAGGLE_DATASET = "your-kaggle-dataset-name"
SCRIPTS_DIR = "scripts"
SRC_DIR = "src"
DATA_DIR = "data"

.PHONY: data merge preprocess pipeline train test format lint check

data:
	@echo "Fetching Kaggle dataset..."
	poetry run python $(SCRIPTS_DIR)/data/download_kaggle.py
	@echo "Running AutoScout24 scraper..."
	poetry run bash scripts/data/scrape_data.sh
	@echo "Data acquisition complete!"

merge:
	@echo "Merging datasets..."
	poetry run python $(SCRIPTS_DIR)/data/merge_data.py
	@echo "Running preprocessing pipeline..."
	poetry run python drafts/run_preprocessing.py

train:
	@echo "Training models and logging to MLflow..."
	poetry run python -m src.models.train --strategy none
	@echo "Training complete! Results in reports/results/"

preprocess:
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

pipeline: merge preprocess test check
	@echo "Full data pipeline executed successfully!"
