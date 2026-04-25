.PHONY: data preprocess pipeline

data:
	@echo "Fetching Kaggle dataset..."
	poetry run python src/data/download_kaggle.py
	@echo "Running AutoScout24 scraper..."
	poetry run python src/data/scrape_autoscout.py # Alaadin please upload ur script
	@echo "Data acquisition complete!"

preprocess:
	@echo "Merging datasets..."
	poetry run python src/data/merge_data.py

pipeline: data preprocess
	@echo "Full data pipeline executed successfully!"