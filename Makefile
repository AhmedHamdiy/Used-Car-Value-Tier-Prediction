# Variables
KAGGLE_DATASET = "your-kaggle-dataset-name"
SCRIPTS_DIR = "scripts"
SRC_DIR = "src"
DATA_DIR = "data"

.PHONY: data preprocess pipeline

data:
	@echo "Fetching Kaggle dataset..."
	poetry run python {SCRIPTS_DIR}/data/download_kaggle.py
	@echo "Running AutoScout24 scraper..."
	poetry run python scripts/data/scrape_autoscout.py # Alaadin please upload ur script
	@echo "Data acquisition complete!"

preprocess:
	@echo "Merging datasets..."
	poetry run python scripts/data/merge_data.py

clean:
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

check: format lint
	@echo "Code quality checks passed!"

pipeline: data preprocess check
	@echo "Full data pipeline executed successfully!"
