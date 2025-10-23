import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

MODELS_DIR = "models"  # Puedes cambiar esta ruta

def main():
    # Leer el archivo preprocesado
    df = pd.read_csv(r"C:\ml25-electronenas\src\ml25\datasets\customer_purchases\customer_purchases_train_preprocessed_for_model.csv")

    # Verificar columna objetivo
    target_column = 'label'
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está en los datos.")

    # Convertir fecha si existe
    if "purchase_timestamp" in df.columns:
        df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"], errors="coerce")
        df["purchase_hour"] = df["purchase_timestamp"].dt.hour
        df["purchase_dayofweek"] = df["purchase_timestamp"].dt.dayofweek
        df = df.drop(columns=["purchase_timestamp"])

    # Separar X e y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Dividir en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    # Evaluar
    predicciones = modelo.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, predicciones))
    print("Precision:", precision_score(y_val, predicciones, zero_division=0))
    print("Recall:", recall_score(y_val, predicciones, zero_division=0))
    print("F1 Score:", f1_score(y_val, predicciones, zero_division=0))
    print("Confusión:", confusion_matrix(y_val, predicciones))
    print("Reporte:", classification_report(y_val, predicciones, zero_division=0))

    # Guardar modelo
    filepath = Path(r"C:\ml25-electronenas\src\ml25\P01_customer_purchases\modelo_rf.pkl")
    joblib.dump(modelo, filepath)
    print(f"Modelo guardado en: {filepath}")

def load_model(filename: str):
    filepath = Path(r"C:\ml25-electronenas\src\ml25\P01_customer_purchases\modelo_rf.pkl")

    modelo = joblib.load(filepath)
    print(f"Modelo cargado desde: {filepath}")
    return modelo

if __name__ == "__main__":
    main()