import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Configuración general ---
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = (CURRENT_FILE / "../../../datasets/customer_purchases/").resolve()

# --- Funciones base ---
def read_csv(filename: str):
    # Quita extensión duplicada si el usuario pasa ".csv"
    if filename.endswith(".csv"):
        filename = filename[:-4]
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    if not os.path.exists(fullfilename):
        raise FileNotFoundError(f"❌ Archivo no encontrado: {fullfilename}")
    df = pd.read_csv(fullfilename)
    return df

def save_df(df, filename: str):
    if not filename.endswith(".csv"):
        filename += ".csv"
    save_path = os.path.join(DATA_DIR, filename)
    df.to_csv(save_path, index=False)
    print(f"✅ df guardado en: {save_path}")

# --- Generación de negativos ---
def get_negatives(df):
    unique_customers = df["customer_id"].unique()
    unique_items = set(df["item_id"].unique())
    negatives = {}
    for customer in unique_customers:
        purchased_items = df[df["customer_id"] == customer]["item_id"].unique()
        non_purchased = unique_items - set(purchased_items)
        negatives[customer] = non_purchased
    return negatives

def gen_all_negatives(df):
    negatives = get_negatives(df)
    negative_lst = []
    for customer_id, item_set in negatives.items():
        negatives_for_customer = [
            {"customer_id": customer_id, "item_id": item_id, "label": 0}
            for item_id in item_set
        ]
        negative_lst.extend(negatives_for_customer)
    return pd.DataFrame(negative_lst)

def gen_random_negatives(df, n_per_positive=2):
    negatives = get_negatives(df)
    negative_lst = []
    for customer_id, item_set in negatives.items():
        if len(item_set) > 0:
            rand_items = np.random.choice(
                list(item_set),
                size=min(n_per_positive, len(item_set)),
                replace=False
            )
            negatives_for_customer = [
                {"customer_id": customer_id, "item_id": item_id, "label": 0}
                for item_id in rand_items
            ]
            negative_lst.extend(negatives_for_customer)
    return pd.DataFrame(negative_lst)

def gen_final_dataset(train_df, negatives):
    # Si el train no tiene label, le ponemos 1
    if 'label' not in train_df.columns:
        train_df['label'] = 1
    combined = pd.concat([train_df, negatives], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined

# --- Procesamiento y extracción de características ---
def extract_customer_features(train_df):
    customer_feat = train_df.groupby("customer_id").agg({
        "customer_date_of_birth": "first",
        "customer_gender": "first",
        "customer_signup_date": "first"
    }).reset_index()
    save_df(customer_feat, "customer_features")
    return customer_feat

def process_df(df, training=True):
    # Placeholder para tu pipeline sklearn
    return df

def preprocess(raw_df, training=False):
    processed_df = process_df(raw_df, training)
    return processed_df

def df_to_numeric(df):
    data = df.copy()
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    return data

def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)
    negatives = gen_random_negatives(train_df, n_per_positive=3)
    final_df = gen_final_dataset(train_df, negatives)
    return final_df

def read_test_data():
    test_df = read_csv("customer_purchases_test")
    # Leer correctamente sin doble extensión
    customer_feat = read_csv("customer_features")
    X_test = test_df
    return X_test

if __name__ == "__main__":
    # Leer y procesar datos
    train_df = read_train_data()
    test_df = read_test_data()

    # Guardar resumen en archivo
    with open(os.path.join(DATA_DIR, "data_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== TRAIN DATA INFO ===\n")
        train_df.info(buf=f)
        f.write("\n\n=== TEST DATA COLUMNS ===\n")
        f.write(", ".join(test_df.columns))
