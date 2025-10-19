from datetime import datetime
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import resample
from pathlib import Path
import os

# Configuración de rutas
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = CURRENT_FILE / "../../datasets/customer_purchases/"


def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    return pd.read_csv(fullfilename)


def main():
    # --- Cargar datos ---
    df = read_csv("customer_purchases_train_final_preprocessed")

    # --- 2. Verificar la columna objetivo ---
    target_column = "label"
    if target_column not in df.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no está en los datos.")

    # --- 3. Procesar columna de fecha (si existe) ---
    if "purchase_timestamp" in df.columns:
        df["purchase_timestamp"] = pd.to_datetime(df["purchase_timestamp"], errors="coerce")
        df["purchase_hour"] = df["purchase_timestamp"].dt.hour.fillna(-1).astype(int)
        df["purchase_dayofweek"] = df["purchase_timestamp"].dt.dayofweek.fillna(-1).astype(int)
        df = df.drop(columns=["purchase_timestamp"])

    # --- 4. Separar variables predictoras y etiqueta ---
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- 5. Imputar valores faltantes ---
    imputer = SimpleImputer(strategy="mean")
    X_imputado = imputer.fit_transform(X)

    # --- 6. Balanceo de clases (upsampling) ---
    df_bal = pd.concat([pd.DataFrame(X_imputado, columns=X.columns), pd.Series(y, name='label')], axis=1)
    df_majority = df_bal[df_bal['label'] == 1]
    df_minority = df_bal[df_bal['label'] == 0]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    X_bal = df_balanced.drop(columns=['label']).values
    y_bal = df_balanced['label'].values

    # --- 7. Dividir en entrenamiento (70%) y prueba (30%) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    # --- 8. Entrenar modelo KNN ---
    modelo = KNeighborsClassifier(n_neighbors=5)
    modelo.fit(X_train, y_train)

    # --- 9. Predicciones ---
    predicciones = modelo.predict(X_test)

    # --- 10. Evaluación ---
    print("\n📊 Métricas de evaluación:")
    print("Accuracy:", accuracy_score(y_test, predicciones))
    print("Precision (macro):", precision_score(y_test, predicciones, average='macro'))
    print("Recall (macro):", recall_score(y_test, predicciones, average='macro'))
    print("F1 Score (macro):", f1_score(y_test, predicciones, average='macro'))

    print("\n🔍 Matriz de confusión:")
    print(confusion_matrix(y_test, predicciones, labels=[0, 1]))

    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_test, predicciones, labels=[0, 1], zero_division=0))

    # --- 11. Validación cruzada (5 folds estratificados) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo, X_bal, y_bal, cv=skf, scoring='f1_macro')

    print("\n🔁 Validación cruzada (F1 macro, 5 folds):")
    print("Scores individuales:", cv_scores)
    print("Promedio F1 macro:", cv_scores.mean())

    train_score = modelo.score(X_train, y_train)
    test_score = accuracy_score(y_test, predicciones)
    diferencia = train_score - test_score

    if diferencia > 0.05:  # umbral del 5%
        print("⚠️ Posible overfitting detectado.")
    else:
        print("✅ No se detecta overfitting significativo.")

if __name__ == "__main__":
    main()