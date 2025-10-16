import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# -----------------------------
# Configuración de rutas y fecha
# -----------------------------
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()

try:
    CURRENT_FILE = Path(__file__).resolve()
except NameError:
    CURRENT_FILE = Path.cwd()

DATA_DIR = Path(r"C:\ml25-electronenas\src\ml25\datasets\customer_purchases")

# -----------------------------
# Mapeos y stopwords
# -----------------------------
STOPWORDS = {'that', 'with', 'the', 'your', 'for', 'every', 'must', 'have', 'you'}
color_mapping = {'b':0, 'bl':1, 'g':2, 'o':3, 'p':4, 'r':5, 'w':6, 'y':7}
category_mapping = {'dress':1, 'blouse':2, 'skirt':3, 'jacket':4, 'jeans':5, 'shoes':6, 'shirt':7, 't-shirt':8, 'suit':9}
device_mapping = {'mobile':0, 'desktop':1}

# -----------------------------
# Funciones auxiliares
# -----------------------------
def read_csv(filename: str):
    file = DATA_DIR / f"{filename}.csv"
    return pd.read_csv(file)

def calculate_age(birth_date_str, reference_date=DATA_COLLECTED_AT):
    if pd.isna(birth_date_str):
        return -1
    try:
        birth_date = datetime.strptime(str(birth_date_str), '%Y-%m-%d').date()
        age = reference_date.year - birth_date.year - ((reference_date.month, reference_date.day) < (birth_date.month, birth_date.day))
        return age
    except Exception:
        return -1

def calculate_years_since_release(release_date_str, reference_date=DATA_COLLECTED_AT):
    if pd.isna(release_date_str):
        return 0
    try:
        release_date = datetime.strptime(str(release_date_str), '%Y-%m-%d').date()
        delta_days = (reference_date - release_date).days
        years_since = delta_days / 365.0
        return max(round(years_since, 2), 0)
    except Exception:
        return 0

def extract_id(id_str, prefix):
    try:
        return int(str(id_str).replace(prefix, ''))
    except Exception:
        return -1

def gen_random_negatives(train_df, n_per_positive=2, random_state=42):
    rng = np.random.default_rng(random_state)
    customers = train_df["customer_id"].unique()
    items = train_df["item_id"].unique()
    negatives = []
    seen_pairs = set()

    for _, row in train_df.iterrows():
        cust = row["customer_id"]
        samples = 0
        while samples < n_per_positive:
            item = rng.choice(items)
            if (cust, item) in seen_pairs or (
                (train_df["customer_id"] == cust) & (train_df["item_id"] == item)
            ).any():
                continue
            negatives.append({"customer_id": cust, "item_id": item, "label": 0})
            seen_pairs.add((cust, item))
            samples += 1

    neg_df = pd.DataFrame(negatives)
    return neg_df

# -----------------------------
# Preprocesamiento principal
# -----------------------------
def preprocess_customer_data(df, is_train=True):
    df_processed = df.copy()

    # Customer
    df_processed['customer_age'] = df_processed['customer_date_of_birth'].apply(calculate_age)
    df_processed['customer_gender'] = df_processed['customer_gender'].replace({'female':1, 'male':0}).fillna(2).astype(int)
    df_processed['customer_id_num'] = df_processed['customer_id'].apply(lambda x: extract_id(x, 'CUST_'))

    # Item
    df_processed['item_id_num'] = df_processed['item_id'].apply(lambda x: extract_id(x, 'ITEM_'))
    df_processed['item_category_num'] = df_processed['item_category'].map(category_mapping).fillna(0).astype(int)
    df_processed['item_color'] = df_processed['item_img_filename'].apply(
        lambda x: color_mapping.get(str(x).split(".")[0].replace("img",""), -1) if pd.notna(x) else -1
    )
    df_processed['item_years_since_release'] = df_processed['item_release_date'].apply(calculate_years_since_release)

    # Purchase
    df_processed['purchase_device_num'] = df_processed['purchase_device'].map(device_mapping).fillna(-1).astype(int)

    # Columnas finales
    columns_final = [
        'customer_id_num', 'customer_age', 'customer_gender',
        'item_id_num', 'item_category_num', 'item_price', 'item_color', 'item_avg_rating',
        'item_years_since_release', 'customer_item_views', 'purchase_device_num'
    ]

    if is_train:
        if 'label' in df_processed.columns:
            df_processed['label'] = df_processed['label'].fillna(-1).astype(int)
        else:
            df_processed['label'] = -1
        columns_final.append('label')

    df_processed = df_processed[columns_final]
    return df_processed

# -----------------------------
# Ejecución principal
# -----------------------------
if __name__ == "__main__":
    # Train positivos
    df_train_pos = read_csv("customer_purchases_train")

    # Generar negativos y completar columnas faltantes
    df_train_neg = gen_random_negatives(df_train_pos, n_per_positive=2)
    df_train_neg = df_train_neg.merge(df_train_pos.drop(columns=['label']), on=['customer_id', 'item_id'], how='left')

    # Combinar positivos y negativos
    df_train_combined = pd.concat([df_train_pos, df_train_neg], ignore_index=True)

    # Preprocesar
    df_train_processed = preprocess_customer_data(df_train_combined, is_train=True)
    train_output_path = DATA_DIR / "customer_purchases_train_preprocessed_for_model.csv"
    df_train_processed.to_csv(train_output_path, index=False)
    print(f" Train preprocesado con negativos guardado en: {train_output_path}")

    # Test (sin negativos)
    df_test = read_csv("customer_purchases_test")
    df_test_processed = preprocess_customer_data(df_test, is_train=False)
    test_output_path = DATA_DIR / "customer_purchases_test_preprocessed_for_model.csv"
    df_test_processed.to_csv(test_output_path, index=False)
    print(f" Test preprocesado guardado en: {test_output_path}")