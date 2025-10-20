import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from pathlib import Path

def main():
    # 1. Leer el archivo preprocesado
    file_path = Path("C:/Users/Abiga/OneDrive/Documents/Aprendizaje de maquina/ml25-electronenas/src/ml25/datasets/customer_purchases/customer_purchases_train_final_preprocessed.csv")
    df = pd.read_csv(file_path)
    print(f"✅ Archivo cargado: {file_path}")
    print(f"Forma del dataset: {df.shape}")

    # 2. Verificar que la columna objetivo exista
    target_column = 'label'
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está en los datos.")

    # 3. Convertir columna de fecha en variables numéricas
    if "purchase_timestamp" in df.columns:
        df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"], errors="coerce")
        df["purchase_hour"] = df["purchase_timestamp"].dt.hour.fillna(-1).astype(int)
        df["purchase_dayofweek"] = df["purchase_timestamp"].dt.dayofweek.fillna(-1).astype(int)
        df = df.drop(columns=["purchase_timestamp"])

    # 4. Separar variables predictoras y la etiqueta
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 5. Imputar valores faltantes con la media
    imputer = SimpleImputer(strategy="mean")
    X_imputado = imputer.fit_transform(X)

    # 6. Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_imputado, y, test_size=0.3, random_state=42)

    # 7. Entrenar el modelo KNN
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_train, y_train)

    # 8. Predicciones
    predicciones = modelo.predict(X_test)

    # 9. Evaluación
    print("\n📊 Métricas de evaluación:")
    print("Accuracy:", accuracy_score(y_test, predicciones))
    print("Precision:", precision_score(y_test, predicciones, zero_division=0))
    print("Recall:", recall_score(y_test, predicciones, zero_division=0))
    print("F1 Score:", f1_score(y_test, predicciones, zero_division=0))

    print("\n🔍 Matriz de confusión:")
    print(confusion_matrix(y_test, predicciones, labels=[0, 1]))

    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_test, predicciones, labels=[0, 1], zero_division=0))

if __name__ == "__main__":
    main()