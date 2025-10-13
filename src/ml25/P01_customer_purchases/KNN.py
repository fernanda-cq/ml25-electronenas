import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def main():
    # 1. Leer el archivo preprocesado
    df = pd.read_csv("C:/Users/tania/OneDrive/Documents/ML/ml25-electronenas/src/ml25/datasets/customer_purchases/customer_purchases_train_preprocessed.csv")

    # 2. Verificar que la columna objetivo exista
    target_column = 'label'
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está en los datos.")

    # 3. Convertir columna de fecha en variables numéricas ANTES de separar X
    if "purchase_timestamp" in df.columns:
        df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"], errors="coerce")
        df["purchase_hour"] = df["purchase_timestamp"].dt.hour
        df["purchase_dayofweek"] = df["purchase_timestamp"].dt.dayofweek
        df = df.drop(columns=["purchase_timestamp"])

    # 4. Separar variables predictoras y la etiqueta
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 5. Imputar valores faltantes con la media
    imputer = SimpleImputer(strategy="mean")
    X_imputado = imputer.fit_transform(X)

    # 6. Entrenar el modelo KNN
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_imputado, y)

    # 7. Predicciones
    predicciones = modelo.predict(X_imputado)

    # 8. Evaluación
    print("Métricas de evaluación:")
    print("Accuracy:", accuracy_score(y, predicciones))
    print("Precision:", precision_score(y, predicciones, zero_division=0))
    print("Recall:", recall_score(y, predicciones, zero_division=0))
    print("F1 Score:", f1_score(y, predicciones, zero_division=0))

    print("\nMatriz de confusión:")
    print(confusion_matrix(y, predicciones, labels=[0, 1]))

    print("\nReporte de clasificación:")
    print(classification_report(y, predicciones, labels=[0, 1], zero_division=0))

if __name__ == "__main__":
    main()
