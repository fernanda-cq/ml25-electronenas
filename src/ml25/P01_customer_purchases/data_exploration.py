import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import json

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../datasets/customer_purchases/"


def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    df = pd.read_csv(fullfilename)
    return df


if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    print(train_df.info())
    test_df = read_csv("customer_purchases_test")
    print(test_df.columns)

    titulos_categorias_unicos = train_df['item_category'].unique()
    print("categorias únicos:")
    print(titulos_categorias_unicos)
    print(f"Total de categorias únicos: {len(titulos_categorias_unicos)}")

    titulos_prendas_unicos = train_df['item_title'].unique()
    print("Títulos únicos:")
    print(titulos_prendas_unicos)
    print(f"Total de títulos únicos: {len(titulos_prendas_unicos)}")
    
    unique_titles_sorted = sorted(titulos_prendas_unicos)  # orden alfabético
    title_mapping = {title: i+1 for i, title in enumerate(unique_titles_sorted)}

    with open(DATA_DIR / "item_title_mapping.txt", "w", encoding="utf-8") as f:
        for title, num in title_mapping.items():
            f.write(f"{num}. {title}\n")

    titulos_devices = train_df['purchase_device'].unique()
    print("Títulos únicos:")
    print(titulos_devices)
    print(f"Total de devices únicos: {len(titulos_devices)}")

