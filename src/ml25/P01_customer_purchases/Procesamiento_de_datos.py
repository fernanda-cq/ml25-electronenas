import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import os

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../datasets/customer_purchases/"

def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
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

color_mapping = {'b':0, 'bl':1, 'g':2, 'o':3, 'p':4, 'r':5, 'w':6, 'y':7}
category_mapping = {'dress':1, 'blouse':2, 'skirt':3, 'jacket':4, 'jeans':5, 'shoes':6, 'shirt':7, 't-shirt':8, 'suit':9}
device_mapping = {'mobile':0, 'desktop':1,}

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
    df_processed['item_category_num'] = df_processed['item_category'].map(category_mapping)
    df_processed['item_years_since_release'] = df_processed['item_release_date'].apply(calculate_years_since_release)
    if 'purchase_device' in df_processed.columns:
        df_processed['purchase_device_num'] = df_processed['purchase_device'].map(device_mapping).fillna(-1).astype(int)
    
    unique_titles = df_processed['item_title'].unique()
    title_mapping = {title: i+1 for i, title in enumerate(unique_titles)}  
    df_processed['item_title_num'] = df_processed['item_title'].map(title_mapping)

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

if __name__ == "__main__":
    df = read_csv("customer_purchases_train") 
    df_processed = preprocess_customer_data(df)
 
    print("Primeras filas del dataset preprocesado:")
    print(df_processed.head())
    print(f"\nForma del dataset: {df_processed.shape}")
    print(f"\nColumnas finales: {df_processed.columns.tolist()}")
    
    output_path = DATA_DIR / "customer_purchases_train_preprocessed.csv"
    df_processed.to_csv(output_path, index=False)
    print(f"\nDataset preprocesado guardado en: {output_path}")

