import kagglehub
import shutil
import os

dataset_name = "thedevastator/uncovering-factors-that-affect-used-car-prices"
cache_path = kagglehub.dataset_download(dataset_name)
target_path = "./data/raw"

if os.path.exists(target_path):
    shutil.rmtree(target_path)
shutil.copytree(cache_path, target_path)

print("Dataset successfully moved to:", target_path)
