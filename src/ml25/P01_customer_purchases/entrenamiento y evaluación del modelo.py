import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_COLLECTED_AT = datetime(2025, 9, 21).date()


CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE.parent.parent / "datasets" / "Procesamiento_de_datos"


def read_csv(filename: str) -> pd.DataFrame:
    file_path = DATA_DIR / f"{filename}.csv"
    return pd.read_csv(file_path)


def main():
    # Datos
    train_df = read_csv("Procesamiento_de_datos")
    test_df = read_csv("Procesamiento_de_datos")

    print("Información del conjunto de entrenamiento:")
    print(train_df.info())
    print("\nColumnas del conjunto de prueba:")
    print(test_df.columns)

    # Convercion 
    df = pd.get_dummies(train_df, drop_first=True)

    # Separa variables
    target_column = 'label'
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está en los datos.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # El modelo
    modelo = RandomForestClassifier(random_state=42)

    
    modelo.fit(X, y)

    # Realiza predicciones
    predicciones = modelo.predict(X)

    
    print("\nPredicciones del modelo:")
    print(predicciones)

    # Evalua que tal lo hizo el modelo 
    print("\nMétricas de evaluación:")
    print("Accuracy:", accuracy_score(y, predicciones))
    print("Precision:", precision_score(y, predicciones))
    print("Recall:", recall_score(y, predicciones))
    print("F1 Score:", f1_score(y, predicciones))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y, predicciones))

    print("\nReporte de clasificación:")
    print(classification_report(y, predicciones))

main ()
    
