import pandas as pd
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

def main():
    # 1. Leer el archivo preprocesado de entrenamiento
    df = pd.read_csv(r"C:\ml25-electronenas\src\ml25\datasets\customer_purchases\customer_purchases_train_preprocessed_for_model.csv")

    # 2. Verificar que la columna objetivo exista
    target_column = 'label'
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está en los datos.")

    # 3. Separar variables predictoras y la etiqueta
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 4. Dividir en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Entrenar el modelo
    modelo = RandomForestClassifier(random_state=42)
    modelo.fit(X_train, y_train)

    # 6. Predicciones sobre el conjunto de validación
    predicciones = modelo.predict(X_val)

    # 7. Evaluación
    print("\n🔍 Métricas de evaluación:")
    print("Accuracy:", accuracy_score(y_val, predicciones))
    print("Precision:", precision_score(y_val, predicciones, zero_division=0))
    print("Recall:", recall_score(y_val, predicciones, zero_division=0))
    print("F1 Score:", f1_score(y_val, predicciones, zero_division=0))

    print("\n📊 Matriz de confusión:")
    print(confusion_matrix(y_val, predicciones, labels=[0, 1]))

    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_val, predicciones, labels=[0, 1], zero_division=0))

if __name__ == "__main__":
    main()