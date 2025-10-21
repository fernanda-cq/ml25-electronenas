import sys
from pathlib import Path
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Paths
# -----------------------------
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../datasets/customer_purchases/"

MODEL_PATH = DATA_DIR / "random_forest_customer_purchases.pkl"
TRAIN_DATA_PATH = DATA_DIR / "customer_purchases_train_with_ids.csv"
OUTPUT_PATH = DATA_DIR / "producto_personalizado_predictions.csv"

def load_model():
    """Cargar modelo entrenado"""
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modelo cargado: {type(model).__name__}")
    return model

def get_feature_order_from_model(model):
    """Obtener orden de features del modelo"""
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_.tolist()
    else:
        return model.estimators_[0].feature_names_in_.tolist()

def define_your_product():
    """
    TÚ defines las características exactas del producto
    """
    print("🎯 DEFINE TU PRODUCTO PERSONALIZADO")
    print("=" * 50)
    
    # PRODUCTO BASE - MODIFICA ESTOS VALORES
    product_features = {
        # PRECIO (cambia este valor)
        'item_price': 500.00,
        
        # CATEGORÍA (cambia la categoría principal a 1, otras a 0)
        'item_category_skirt': 0,      # Falda
        'item_category_dress': 0,      # Vestido
        'item_category_blouse': 0,     # Blusa
        'item_category_jeans': 1,      # Jeans
        'item_category_jacket': 0,     # Chaqueta
        'item_category_shirt': 0,      # Camisa
        'item_category_shoes': 0,      # Zapatos
        'item_category_slacks': 0,     # Pantalones formales
        'item_category_suit': 0,       # Traje
        'item_category_t-shirt': 0,    # Camiseta
        
        # IMAGEN (elige una)
        'item_img_filename_imgr.jpg': 0,    # roja
        'item_img_filename_imgw.jpg': 0,    # blanca
        'item_img_filename_imgo.jpg': 0,    # naranja
        'item_img_filename_imgp.jpg': 0,    # púrpura
        'item_img_filename_imgb.jpg': 1,    # azul
        'item_img_filename_imgg.jpg': 0,    # verde
        'item_img_filename_imgbl.jpg': 0,   # negra
        'item_img_filename_imgy.jpg': 0,    # amarilla
        
        # PALABRAS CLAVE (activa las que quieras)
        'item_title_bow_elegant': 0,    # Elegante
        'item_title_bow_modern': 1,     # Moderno
        'item_title_bow_classic': 1,    # Clásico
        'item_title_bow_casual': 1,     # Casual
        'item_title_bow_premium': 0,    # Premium
        'item_title_bow_exclusive': 0,  # Exclusivo
        'item_title_bow_red': 0,        # Rojo
        'item_title_bow_blue': 0,       # Azul
        'item_title_bow_black': 0,      # Negro
        'item_title_bow_white': 0,      # Blanco
        'item_title_bow_special': 1,    # Especial
        'item_title_bow_occasion': 1,   # Para ocasiones
        'item_title_bow_collection': 0, # Colección
        'item_title_bow_lightweight': 0,# Ligero
        'item_title_bow_durable': 1,    # Duradero
        'item_title_bow_stylish': 1,    # Estiloso
    }
    
    # MOSTRAR RESUMEN DEL PRODUCTO
    print("\n📦 TU PRODUCTO DEFINIDO:")
    print(f"   💰 Precio: ${product_features['item_price']}")
    
    # Mostrar categoría activa
    categories = [k for k, v in product_features.items() if k.startswith('item_category_') and v == 1]
    if categories:
        category_name = categories[0].replace('item_category_', '').upper()
        print(f"   📂 Categoría: {category_name}")
    
    # Mostrar imagen activa
    images = [k for k, v in product_features.items() if k.startswith('item_img_filename_') and v == 1]
    if images:
        image_name = images[0].replace('item_img_filename_', '').upper()
        print(f"   🖼️  Imagen: {image_name}")
    
    # Mostrar palabras clave activas
    active_words = [k.replace('item_title_bow_', '') for k, v in product_features.items() 
                   if k.startswith('item_title_bow_') and v == 1]
    print(f"   🔤 Palabras clave: {', '.join(active_words)}")
    
    print("=" * 50)
    
    return product_features

def predict_for_custom_product():
    """
    Predecir para el producto que TÚ defines - VERSIÓN CORREGIDA
    """
    print("🎯 PREDICCIÓN PARA PRODUCTO PERSONALIZADO")
    
    # 1. Cargar modelo
    model = load_model()
    
    # 2. Obtener orden de features
    feature_order = get_feature_order_from_model(model)
    
    # 3. TÚ defines el producto
    product_features = define_your_product()
    
    # 4. Cargar clientes reales - VERSIÓN CORREGIDA
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        print(f"✅ Datos cargados: {train_df.shape}")
        
        # ✅ VERIFICAR QUÉ COLUMNAS EXISTEN
        print(f"🔍 Columnas disponibles: {train_df.columns.tolist()}")
        
        # Buscar la columna de ID correcta
        if 'purchase_id' in train_df.columns:
            id_column = 'purchase_id'
            print("✅ Usando 'purchase_id' como identificador")
        elif 'ID' in train_df.columns:
            id_column = 'ID'
            print("✅ Usando 'ID' como identificador") 
        elif 'customer_id' in train_df.columns:
            id_column = 'customer_id'
            print("✅ Usando 'customer_id' como identificador")
        else:
            # Usar la primera columna que no sea numérica
            non_numeric_cols = train_df.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                id_column = non_numeric_cols[0]
                print(f"✅ Usando '{id_column}' como identificador")
            else:
                id_column = train_df.columns[0]
                print(f"⚠️  Usando primera columna '{id_column}' como identificador")
        
        unique_customers = train_df[[id_column]].drop_duplicates().head(300)
        print(f"✅ Analizando {len(unique_customers)} clientes reales")
        
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None, None
    
    # 5. Para cada cliente: sus features + tu producto
    features_list = []
    customer_info = []
    
    for _, customer_row in unique_customers.iterrows():
        customer_id = customer_row[id_column]
        
        # Encontrar cliente en datos reales
        customer_data = train_df[train_df[id_column] == customer_id]
        if len(customer_data) == 0:
            continue
            
        customer_sample = customer_data.iloc[0]  # Usar primer registro, no aleatorio
        
        # Crear features combinadas
        features = {col: 0 for col in feature_order}  # Inicializar en 0
        
        # A. Features de TU PRODUCTO (fijas)
        for feature, value in product_features.items():
            if feature in feature_order:
                features[feature] = value
        
        # B. Features del CLIENTE (reales)
        for col in ['customer_age_years', 'customer_tenure_years']:
            if col in feature_order and col in customer_sample:
                features[col] = customer_sample[col]
        
        # C. Gender del cliente (real)
        for gender_col in ['customer_gender_female', 'customer_gender_male', 'customer_gender_nan']:
            if gender_col in feature_order and gender_col in customer_sample:
                features[gender_col] = customer_sample[gender_col]
        
        # D. ¿Le gusta esta categoría?
        active_category = [k for k, v in product_features.items() 
                          if k.startswith('item_category_') and v == 1][0]
        category_name = active_category.replace('item_category_', '')
        prefers_this_category = 1 if customer_sample.get('customer_prefered_cat', '') == category_name else 0
        features['customer_cat_is_prefered'] = prefers_this_category
        
        features_list.append(features)
        customer_info.append({
            'customer_id': customer_id,
            'age': customer_sample.get('customer_age_years', 'N/A'),
            'prefers_this_category': prefers_this_category,
            'preferred_category': customer_sample.get('customer_prefered_cat', 'N/A')
        })
    
    # 6. Crear DataFrame y predecir
    X_new = pd.DataFrame(features_list, columns=feature_order)
    
    print(f"✅ Features preparadas: {X_new.shape}")
    
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]
    
    # 7. Resultados
    results = pd.DataFrame({
        'customer_id': [info['customer_id'] for info in customer_info],
        'age': [info['age'] for info in customer_info],
        'prefers_this_category': [info['prefers_this_category'] for info in customer_info],
        'preferred_category': [info['preferred_category'] for info in customer_info],
        'prediction': predictions,
        'probability': probabilities,
        'interest_level': ['ALTO' if p > 0.7 else 'MEDIO' if p > 0.4 else 'BAJO' for p in probabilities]
    })
    
    results = results.sort_values('probability', ascending=False)
    results.to_csv(OUTPUT_PATH, index=False)
    
    print(f"💾 Resultados guardados: {OUTPUT_PATH}")
    
    # 8. ANÁLISIS
    print(f"\n📊 RESULTADOS PARA TU PRODUCTO:")
    total = len(results)
    high_interest = len(results[results['probability'] > 0.7])
    medium_interest = len(results[(results['probability'] > 0.4) & (results['probability'] <= 0.7)])
    low_interest = len(results[results['probability'] <= 0.4])
    
    print(f"   - Total clientes: {total}")
    print(f"   - Alto interés (>0.7): {high_interest} ({high_interest/total*100:.1f}%)")
    print(f"   - Medio interés (0.4-0.7): {medium_interest} ({medium_interest/total*100:.1f}%)")
    print(f"   - Bajo interés (≤0.4): {low_interest} ({low_interest/total*100:.1f}%)")
    
    # Clientes que PREFIEREN esta categoría
    category_lovers = results[results['prefers_this_category'] == 1]
    if len(category_lovers) > 0:
        print(f"\n🔍 Clientes que PREFIEREN esta categoría:")
        print(f"   - Cantidad: {len(category_lovers)}")
        print(f"   - Probabilidad promedio: {category_lovers['probability'].mean():.3f}")
        print(f"   - Alto interés: {len(category_lovers[category_lovers['probability'] > 0.7])}")
    
    # Top clientes
    print(f"\n🏆 TOP 10 CLIENTES MÁS PROPENSOS:")
    top_10 = results.head(10)
    for idx, row in top_10.iterrows():
        category_icon = "✓" if row['prefers_this_category'] == 1 else "○"
        print(f"   {idx+1:2d}. {category_icon} {row['customer_id']} (edad {row['age']}) - {row['probability']:.3f}")
    
    return results, product_features

if __name__ == "__main__":
    try:
        results, product_config = predict_for_custom_product()
        if results is not None:
            print(f"\n🎉 ¡ANÁLISIS COMPLETADO!")
            good_prospects = len(results[results['probability'] > 0.5])
            print(f"📈 Tienes {good_prospects} clientes potenciales para tu producto")
            print(f"💡 Modifica 'define_your_product()' para probar diferentes productos")
        else:
            print("❌ No se pudieron obtener resultados")
    except Exception as e:
        print(f"❌ Error: {e}")