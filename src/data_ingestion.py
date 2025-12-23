import pandas as pd
from sklearn.datasets import fetch_openml
from pathlib import Path
import os 
import urllib.request

DATA_PATH = Path("data/raw")

def download_data():
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    RAW_FILE = os.path.join(DATA_PATH, "processed.cleveland.data")

    if not os.path.exists(RAW_FILE):
        urllib.request.urlretrieve(DATA_URL, RAW_FILE)
        print("Dataset downloaded from UCI repository")
    else:
        print("Dataset already exists locally")

    # data = fetch_openml(name="heart-disease-uci", version=1, as_frame=True)
    # df = data.frame

    # df.to_csv(DATA_PATH / "heart.csv", index=False)
    # print("Dataset downloaded successfully")

if __name__ == "__main__":
    download_data()
