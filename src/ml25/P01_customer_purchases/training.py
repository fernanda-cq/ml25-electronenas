from datetime import datetime
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../datasets/customer_purchases/"

MODEL_OUTPUT = DATA_DIR / "random_forest_customer_purchases.pkl"
TEST_PRED_OUTPUT = DATA_DIR / "rd_predicciones.csv"

def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    return pd.read_csv(fullfilename)

# Función para detectar y prevenir overfitting
def check_and_prevent_overfitting(model, X_train, X_val, y_train, y_val, X_test=None):
    print("\n" + "="*60)
    print("ANALIZANDO OVERFITTING")
    print("="*60)
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    difference = train_score - val_score
    
    print(f"   -Score en train: {train_score:.4f}")
    print(f"   -Score en validation: {val_score:.4f}")
    print(f"   -Diferencia: {difference:.4f}")
    
    # Evaluación de overfitting
    if difference > 0.05:
        print("  POSIBLE OVERFITTING DETECTADO :O")
        return True
    elif difference > 0.02:
        print("  Pequeña diferencia, monitorear :/")
        return False
    else:
        print("  No se detecta overfitting significativo :D")
        return False

# Función para evaluación robusta
def comprehensive_evaluation(model, X_train, X_val, y_train, y_val):
    print("\n" + "="*60)
    print("EVALUACIÓN COMPLETA DEL MODELO")
    print("="*60)
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    
    # Métricas
    train_accuracy = model.score(X_train, y_train)
    val_accuracy = model.score(X_val, y_val)
    val_roc_auc = roc_auc_score(y_val, y_val_prob)
    
    print(f"    Accuracy - Train: {train_accuracy:.4f}, Val: {val_accuracy:.4f}")
    print(f"    ROC-AUC Validation: {val_roc_auc:.4f}")
    
    # Cross-validation para evaluación más robusta
    cv_scores = cross_val_score(model.best_estimator_, X_train, y_train, 
                               cv=StratifiedKFold(n_splits=5), scoring='f1_macro')
    print(f"   Cross-val F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Reporte de clasificación
    print("\n      Classification Report (Validation):")
    print(classification_report(y_val, y_val_pred))

# Función principal
def main():
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60)
    
    # Leer datos
    train_df = read_csv("customer_purchases_train_with_ids")
    test_df = read_csv("customer_purchases_test_with_ids")

    # ✅ VERIFICACIÓN CRÍTICA: Revisar qué columnas tenemos
    print("🔍 COLUMNAS EN TRAIN:")
    for col in train_df.columns:
        print(f"   - {col} ({train_df[col].dtype})")
    
    # ✅ SOLUCIÓN: ELIMINAR TODAS LAS COLUMNAS QUE PUEDAN CONTENER STRINGS
    columnas_a_eliminar = [
        "customer_id", "purchase_id",  # IDs que contienen strings
        "item_title", "item_img_filename",  # Texto
        "customer_date_of_birth", "customer_signup_date",  # Fechas
        "purchase_timestamp", "item_release_date"  # Más fechas
    ]
    
    # Eliminar columnas problemáticas
    columnas_eliminadas_train = [col for col in columnas_a_eliminar if col in train_df.columns]
    columnas_eliminadas_test = [col for col in columnas_a_eliminar if col in test_df.columns]
    
    print(f"🗑️  Eliminando columnas con strings:")
    print(f"   - Train: {columnas_eliminadas_train}")
    print(f"   - Test: {columnas_eliminadas_test}")
    
    y_train_full = train_df["label"].copy()
    
    # Crear features eliminando columnas problemáticas
    train_df_features = train_df.drop(columns=columnas_eliminadas_train, errors="ignore")
    test_df_features = test_df.drop(columns=columnas_eliminadas_test, errors="ignore")
    
    # ✅ VERIFICACIÓN EXTRA: Solo mantener columnas numéricas
    train_df_features = train_df_features.select_dtypes(include=[np.number])
    test_df_features = test_df_features.select_dtypes(include=[np.number])
    
    print(f"✅ Features después de limpieza:")
    print(f"   - Train: {train_df_features.shape}")
    print(f"   - Test: {test_df_features.shape}")
    
    # Encontrar columnas comunes
    common_cols = list(set(train_df_features.columns) & set(test_df_features.columns))
    
    features_stop_words = [
        'item_title_bow_the', 'item_title_bow_you', 'item_title_bow_your',
        'item_title_bow_that', 'item_title_bow_for', 'item_title_bow_with',
        'item_title_bow_have', 'item_title_bow_must', 'item_title_bow_need',
        'item_title_bow_every', 'item_title_bow_out', 'item_title_bow_up',
        'item_title_bow_occasion', 'item_title_bow_stand', 'item_title_bow_step',
        'item_title_bow_style'
    ]
    
    features_stop_words_existentes = [f for f in features_stop_words if f in common_cols]
    common_cols_filtradas = [col for col in common_cols if col not in features_stop_words_existentes]
    
    print(f"    Eliminando {len(features_stop_words_existentes)} features de stop words")
    print(f"    Columnas originales: {len(common_cols)}")
    print(f"    Columnas después de eliminar stop words: {len(common_cols_filtradas)}")
    
    X_train_full = train_df_features[common_cols_filtradas]
    X_test = test_df_features[common_cols_filtradas]
    
    # ✅ VERIFICACIÓN FINAL: Asegurar que no hay strings
    print("🔍 VERIFICACIÓN FINAL DE TIPOS DE DATOS:")
    for col in X_train_full.columns:
        if X_train_full[col].dtype == 'object':
            print(f"❌ PROBLEMA: Columna {col} todavía es string")
            # Eliminar columnas problemáticas
            X_train_full = X_train_full.drop(columns=[col])
            X_test = X_test.drop(columns=[col], errors='ignore')
    
    print(f"✅ Shape final:")
    print(f"   - X_train_full: {X_train_full.shape}")
    print(f"   - X_test: {X_test.shape}")
    
    # Combinar features y labels para el sampling
    train_data = pd.concat([X_train_full, y_train_full], axis=1)
    class_0 = train_data[train_data['label'] == 0]  # Clase minoritaria
    class_1 = train_data[train_data['label'] == 1]  # Clase mayoritaria
    
    print(f"  Distribución original:")
    print(f"     -Clase 0 (negativos): {len(class_0)} ejemplos")
    print(f"     -Clase 1 (positivos): {len(class_1)} ejemplos")
    print(f"     -Ratio: 1:{len(class_1)//len(class_0)}")
    
    # UNDERSAMPLING: Tomar una muestra de la clase mayoritaria
    undersample_ratio = 2  # 1:2 ratio (clase 0 : clase 1)
    n_samples_class_1 = len(class_0) * undersample_ratio
    
    # Tomar muestra aleatoria de la clase mayoritaria
    class_1_undersampled = class_1.sample(n=min(n_samples_class_1, len(class_1)), 
                                        random_state=42)
    
    # Combinar las clases balanceadas
    balanced_data = pd.concat([class_0, class_1_undersampled], axis=0)
    
    # Mezclar los datos
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separar features y labels nuevamente
    X_train_balanced = balanced_data.drop(columns=['label'])
    y_train_balanced = balanced_data['label']
    
    print(f"  Distribución después de undersampling:")
    print(f"     -Clase 0 (negativos): {len(class_0)} ejemplos")
    print(f"     -Clase 1 (positivos): {len(class_1_undersampled)} ejemplos")
    print(f"     -Ratio: 1:{undersample_ratio}")
    print(f"     -Total ejemplos: {len(balanced_data)}")

    # Verificar consistencia de columnas
    train_cols = set(X_train_balanced.columns)
    test_cols = set(X_test.columns)
    
    if train_cols != test_cols:
        print(" ERROR: Aún hay diferencia en columnas")
        final_common_cols = list(train_cols.intersection(test_cols))
        X_train_balanced = X_train_balanced[final_common_cols]
        X_test = X_test[final_common_cols]
        print(f" Usando {len(final_common_cols)} columnas finales")
    else:
        print("Train y test tienen las mismas columnas :D")

    # Análisis de características disponibles
    print("\n" + "="*60)
    print("   ANALISIS DE CARACTERISTICAS")
    print("="*60)
    customer_features = [col for col in X_train_balanced.columns if 'customer' in col]
    item_features = [col for col in X_train_balanced.columns if 'item' in col and 'customer' not in col]
    bow_features = [col for col in X_train_balanced.columns if 'bow' in col]
    
    print(f"    -Características del cliente: {len(customer_features)}")
    print(f"    -Características del item: {len(item_features)}")
    print(f"    -Características de texto (BOW): {len(bow_features)}")
    print(f"    -Total características: {X_train_balanced.shape[1]}")
    print(f" Distribución de clases después de undersampling: {np.bincount(y_train_balanced)}")
    class_weight_dict = None

    # División train/validation (80/20) del dataset balanceado
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_balanced, y_train_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_balanced  # Mantener proporción en la división
    )

    print(f" Train split balanceado: {X_train.shape}")
    print(f" Validation split balanceado: {X_val.shape}")
    print(f" Distribución en train: {np.bincount(y_train)}")
    print(f" Distribución en validation: {np.bincount(y_val)}")
    
    # MODELO MÁS RESTRICTIVO PARA REDUCIR OVERFITTING
    rf = RandomForestClassifier(
        random_state=42,
        max_depth=8,            #  Más bajo
        min_samples_split=20,   #  Más alto  
        min_samples_leaf=8,     #  Más alto
        n_estimators=100,
        max_features='sqrt'
    )
    
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [2, 4, 6],
        "max_features": ["sqrt", "log2", 0.5],
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=15,
        cv=StratifiedKFold(n_splits=3),
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Entrenar modelo
    print("\n" + "="*60)
    print("USANDO DATOS BALANCEADOS")
    print("="*60)
    
    # ✅ VERIFICACIÓN FINAL ANTES DE ENTRENAR
    print("🔍 VERIFICACIÓN FINAL ANTES DEL ENTRENAMIENTO:")
    print(f"   - X_train shape: {X_train.shape}")
    print(f"   - X_train dtypes: {X_train.dtypes.unique()}")
    print(f"   - ¿Hay strings?: {'object' in X_train.dtypes.astype(str)}")
    
    search.fit(X_train, y_train)
    
    best_rf = search.best_estimator_
    print(f" Mejores hiperparámetros: {search.best_params_}")

    # Evaluación comprehensiva
    metrics = comprehensive_evaluation(search, X_train, X_val, y_train, y_val)

    # Detección de overfitting
    has_overfitting = check_and_prevent_overfitting(best_rf, X_train, X_val, y_train, y_val)
    
    # Si hay overfitting, entrenar modelo más simple
    if has_overfitting:
        print("\n ENTRENANDO MODELO MÁS SIMPLE...")
        simple_rf = RandomForestClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=15,
            min_samples_leaf=6,
            n_estimators=100,
            max_features='sqrt'
        )
        simple_rf.fit(X_train, y_train)
        
        simple_val_score = simple_rf.score(X_val, y_val)
        best_val_score = best_rf.score(X_val, y_val)
        
        print(f"Comparación de modelos:")
        print(f"   Modelo complejo (val): {best_val_score:.4f}")
        print(f"   Modelo simple (val): {simple_val_score:.4f}")
        
        if simple_val_score >= best_val_score * 0.98:
            final_model = simple_rf
            print(" Usando modelo simple (mejor generalización) :/")
        else:
            final_model = best_rf
            print(" Usando modelo optimizado :)")
    else:
        final_model = best_rf
        print(" Usando modelo optimizado :D")

    # Análisis de importancia de características
    print("\n" + "="*60)
    print("   ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS   ")
    print("="*60)
    feature_importance = pd.DataFrame({
        'feature': X_train_balanced.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("   Top 10 características más importantes:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"      {row['feature']}: {row['importance']:.4f}")

    # Validación cruzada final en datos balanceados
    print("\n" + "="*60)
    print("  VALIDACIÓN CRUZADA FINAL (5-fold)   ")
    print("="*60)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_final = cross_val_score(final_model, X_train_balanced, y_train_balanced, 
                                    cv=skf, scoring="roc_auc")
    
    print(f"   ROC-AUC Scores: {np.round(cv_scores_final, 4)}")
    print(f"   Promedio: {cv_scores_final.mean():.4f} ± {cv_scores_final.std():.4f}")

    # Guardar modelo
    joblib.dump(final_model, MODEL_OUTPUT)
    print(f"💾 Modelo guardado en: {MODEL_OUTPUT}")

    # PREDICCIONES PARA KAGGLE
    print("\n" + "="*60)
    print(" PREDICCIONES PARA KAGGLE")
    print("="*60)
    
    # Leer datos de test ORIGINALES para obtener los purchase_id
    original_test = read_csv("customer_purchases_test")
    purchase_ids = original_test['purchase_id'].unique()
    
    # Hacer predicciones
    test_pred = final_model.predict(X_test)
    
    # Crear DataFrame para Kaggle
    kaggle_submission = pd.DataFrame({
        'ID': purchase_ids,
        'pred': test_pred
    })
    
    print(f"✅ Kaggle submission: {len(kaggle_submission)} registros")
    
    # Guardar archivo final para Kaggle
    kaggle_submission.to_csv(TEST_PRED_OUTPUT, index=False)

    
    # Resumen final
    print(f"\n     RESUMEN FINAL:")
    print(f"   - Modelo: Random Forest con undersampling")
    print(f"   - Ratio usado: 1:{undersample_ratio}")
    print(f"   - Características usadas: {X_train_balanced.shape[1]} (sin stop words)")
    print(f"   - Registros en submission: {len(kaggle_submission)}")
    print(f"   - Predicciones positivas (1): {kaggle_submission['pred'].sum()}")
    print(f"   - Predicciones negativas (0): {len(kaggle_submission) - kaggle_submission['pred'].sum()}")
    print(f"   - ROC-AUC CV: {cv_scores_final.mean():.4f}")
    

if __name__ == "__main__":
    main()