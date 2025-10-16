import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Configuración de rutas
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = Path("C:/Users/tania/OneDrive/Documents/ML/ml25-electronenas/src/ml25/datasets/customer_purchases/")

# Mapeos
color_mapping = {'b':0, 'bl':1, 'g':2, 'o':3, 'p':4, 'r':5, 'w':6, 'y':7}
category_mapping = {'dress':1, 'blouse':2, 'skirt':3, 'jacket':4, 'jeans':5, 'shoes':6, 'shirt':7, 't-shirt':8, 'suit':9}
device_mapping = {'mobile':0, 'desktop':1}

# Funciones auxiliares
def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    print(f"Buscando archivo en: {fullfilename}")
    df = pd.read_csv(fullfilename)
    return df

def calculate_age(birth_date_str, reference_date=DATA_COLLECTED_AT):
    birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
    age = reference_date.year - birth_date.year - ((reference_date.month, reference_date.day) < (birth_date.month, birth_date.day))
    return age

def calculate_tenure(signup_date_str, reference_date=DATA_COLLECTED_AT):
    signup_date = datetime.strptime(signup_date_str, '%Y-%m-%d').date()
    tenure = reference_date.year - signup_date.year - ((reference_date.month, reference_date.day) < (signup_date.month, signup_date.day))
    return tenure

def calculate_years_since_release(release_date_str, reference_date=DATA_COLLECTED_AT):
    release_date = datetime.strptime(release_date_str, '%Y-%m-%d').date()
    delta_days = (reference_date - release_date).days
    years_since = delta_days / 365.0
    return round(years_since, 2)

def extract_id(id_str, prefix):
    return int(id_str.replace(prefix, ''))

def preprocess_customer_data(df):
    df_processed = df.copy()
    df_processed['customer_age'] = df_processed['customer_date_of_birth'].apply(calculate_age)
    df_processed['customer_gender'] = df_processed['customer_gender'].replace({'female': 1,'male': 0}).fillna(2).astype(int)
    df_processed['customer_tenure'] = df_processed['customer_signup_date'].apply(calculate_tenure)
    df_processed['customer_id_num'] = df_processed['customer_id'].apply(lambda x: extract_id(x, 'CUST_'))
    df_processed['item_id_num'] = df_processed['item_id'].apply(lambda x: extract_id(x, 'ITEM_'))

    def extract_img_color(img_name):
        if pd.isna(img_name):
            return -1
        base = img_name.split(".")[0]
        letters = base.replace("img","")
        return color_mapping.get(letters, -1)

    df_processed['item_color'] = df_processed['item_img_filename'].apply(extract_img_color)
    df_processed['item_category_num'] = df_processed['item_category'].map(category_mapping).fillna(-1).astype(int)
    df_processed['item_years_since_release'] = df_processed['item_release_date'].apply(calculate_years_since_release)

    if 'purchase_device' in df_processed.columns:
        df_processed['purchase_device_num'] = df_processed['purchase_device'].map(device_mapping).fillna(-1).astype(int)

    unique_titles = df_processed['item_title'].unique()
    title_mapping = {title: i+1 for i, title in enumerate(unique_titles)}
    df_processed['item_title_num'] = df_processed['item_title'].map(title_mapping).fillna(-1).astype(int)

    df_processed['item_price'] = df_processed['item_price'].apply(lambda x: x if pd.notna(x) and x >= 0 else -1)
    df_processed['item_avg_rating'] = df_processed['item_avg_rating'].fillna(-1)
    df_processed['item_num_ratings'] = df_processed['item_num_ratings'].fillna(-1).astype(int)
    df_processed['customer_item_views'] = df_processed['customer_item_views'].apply(lambda x: x if pd.notna(x) and x >= 0 else -1)

    columns_order = [
        'purchase_id', 'customer_id_num', 'customer_age', 'customer_gender', 'customer_signup_date',
        'item_id_num', 'item_title_num', 'item_category_num', 'item_price', 'item_color',
        'item_avg_rating', 'item_num_ratings', 'item_years_since_release', 'purchase_timestamp',
        'customer_item_views', 'purchase_item_rating', 'purchase_device_num', 'label'
    ]

    columns_existing = [col for col in columns_order if col in df_processed.columns]
    df_processed = df_processed[columns_existing]

    columns_to_drop = ['customer_date_of_birth','customer_signup_date','item_img_filename','item_release_date']
    columns_to_drop_existing = [col for col in columns_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=columns_to_drop_existing)

    return df_processed

# Generación de negativos
def get_negatives(df):
    unique_customers = df["customer_id"].unique()
    unique_items = set(df["item_id"].unique())

    negatives = {}
    for customer in unique_customers:
        purchased_items = df[df["customer_id"] == customer]["item_id"].unique()
        non_purchased = unique_items - set(purchased_items)
        negatives[customer] = non_purchased
    return negatives

def gen_random_negatives(df, n_per_positive=2):
    negatives = get_negatives(df)
    negative_lst = []
    for customer_id, item_set in negatives.items():
        if len(item_set) >= n_per_positive:
            rand_items = np.random.choice(list(item_set), size=n_per_positive, replace=False)
        else:
            rand_items = list(item_set)
        negatives_for_customer = [
            {"customer_id": customer_id, "item_id": item_id, "label": 0}
            for item_id in rand_items
        ]
        negative_lst.extend(negatives_for_customer)
    return pd.DataFrame(negative_lst)

def enrich_negatives(negatives, train_df):
    customer_cols = ['customer_id', 'customer_date_of_birth', 'customer_gender', 'customer_signup_date']
    item_cols = ['item_id', 'item_title', 'item_category', 'item_price', 'item_img_filename',
                 'item_avg_rating', 'item_num_ratings', 'item_release_date']

    customer_info = train_df[customer_cols].drop_duplicates()
    item_info = train_df[item_cols].drop_duplicates()

    enriched = negatives.merge(customer_info, on='customer_id', how='left')
    enriched = enriched.merge(item_info, on='item_id', how='left')

    enriched['purchase_id'] = np.nan
    enriched['purchase_timestamp'] = np.nan
    enriched['customer_item_views'] = -1
    enriched['purchase_item_rating'] = -1
    enriched['purchase_device'] = np.nan

    return enriched

def gen_final_dataset(train_df, negatives):
    train_df_labeled = train_df.copy()
    if "label" in train_df_labeled.columns:
        train_df_labeled = train_df_labeled.drop(columns=["label"])
    train_df_labeled["label"] = 1

    negatives_cleaned = negatives.copy()
    negatives_cleaned = negatives_cleaned.loc[:, ~negatives_cleaned.columns.duplicated()]
    negatives_cleaned["label"] = 0

    common_cols = list(set(train_df_labeled.columns) & set(negatives_cleaned.columns))
    common_cols.append("label")

    train_df_filtered = train_df_labeled[common_cols]
    negatives_filtered = negatives_cleaned[common_cols]

    final_df = pd.concat([train_df_filtered, negatives_filtered], ignore_index=True)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    return final_df

# Ejecución principal
if __name__ == "__main__":
    train_df = read_csv("customer_purchases_train")
    randnegatives = gen_random_negatives(train_df, n_per_positive=3)
    enriched_negatives = enrich_negatives(randnegatives, train_df)
    final_dataset = gen_final_dataset(train_df, enriched_negatives)

    print("\nDataset combinado (positivos + negativos):")
    print(final_dataset.head())
    print(f"\nForma: {final_dataset.shape}")
    print(f"\nColumnas únicas: {final_dataset.columns.tolist()}")
    print(f"\nDistribución de etiquetas:\n{final_dataset['label'].value_counts()}")

    df_processed = preprocess_customer_data(final_dataset)
    print("\nPrimeras filas del dataset preprocesado:")
    print(df_processed.head())
    print(f"\nForma final: {df_processed.shape}")
    print(f"\nColumnas finales: {df_processed.columns.tolist()}")

    # Guardar
    output_path = DATA_DIR / "customer_purchases_train_final_preprocessed.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"\n✅ Dataset final preprocesado guardado en: {output_path}")