from datetime import datetime
import os
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# -----------------------------
# Configuración de rutas y fecha
# -----------------------------
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = (CURRENT_FILE / "../../datasets/customer_purchases/").resolve()


# -----------------------------
# Función para leer CSV
# -----------------------------
def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    df = pd.read_csv(file)
    return df


# -----------------------------
# Función principal
# -----------------------------
def main():
    # Cargar datos preprocesados
    df = read_csv("customer_purchases_train_final_preprocessed")

    target_column = 'label'
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está en los datos.")

    # Convertir fechas si existen
    if "purchase_timestamp" in df.columns:
        df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"], errors="coerce")
        df["purchase_hour"] = df["purchase_timestamp"].dt.hour
        df["purchase_dayofweek"] = df["purchase_timestamp"].dt.dayofweek
        df = df.drop(columns=["purchase_timestamp"])

    # Separar variables
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # División en entrenamiento y prueba (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modelo ajustado con limitaciones para evitar sobreajuste
    modelo = DecisionTreeClassifier(max_depth=6, min_samples_split=30, random_state=42)
    modelo.fit(X_train, y_train)

    # Predicciones sobre datos de prueba
    predicciones = modelo.predict(X_test)

    # -----------------------------
    # Evaluación del modelo
    # -----------------------------
    print("=" * 60)
    print("🔍 RESULTADOS DE EVALUACIÓN (Conjunto de Prueba)")
    print("=" * 60)
    print(f"Accuracy:  {accuracy_score(y_test, predicciones):.4f}")
    print(f"Precision: {precision_score(y_test, predicciones, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predicciones, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, predicciones, zero_division=0):.4f}")

    print("\n📊 Matriz de confusión:")
    print(confusion_matrix(y_test, predicciones, labels=[0, 1]))

    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_test, predicciones, labels=[0, 1], zero_division=0))

    # -----------------------------
    # Validación cruzada opcional
    # -----------------------------
    print("=" * 60)
    print("🔁 VALIDACIÓN CRUZADA (5 Folds)")
    print("=" * 60)
    scores = cross_val_score(modelo, X, y, cv=5)
    print("Scores individuales:", [f"{s:.4f}" for s in scores])
    print(f"Promedio de accuracy (CV): {scores.mean():.4f}")

    # Importancia de variables
    print("=" * 60)
    print("🌳 IMPORTANCIA DE VARIABLES")
    print("=" * 60)
    importances = pd.Series(modelo.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False).head(10))

    pred_train = modelo.predict(X_train)

    print("=" * 60)
    print("🔍 RESULTADOS DE EVALUACIÓN (Conjunto de Entrenamiento)")
    print("=" * 60)
    print(f"Accuracy (train):  {accuracy_score(y_train, pred_train):.4f}")
    print(f"Precision (train): {precision_score(y_train, pred_train, zero_division=0):.4f}")
    print(f"Recall (train):    {recall_score(y_train, pred_train, zero_division=0):.4f}")
    print(f"F1 Score (train):  {f1_score(y_train, pred_train, zero_division=0):.4f}")

# -----------------------------
# Ejecución del script
# -----------------------------
if __name__ == "__main__":
    main()
