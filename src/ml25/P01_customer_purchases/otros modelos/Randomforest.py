# ============================================================
# 🌲 Entrenamiento y evaluación de Random Forest
# con visualizaciones automáticas
# ============================================================

from datetime import datetime
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Paths
# -----------------------------
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = Path("C:/Users/tania/OneDrive/Documents/ML/ml25-electronenas/src/ml25/datasets/customer_purchases")

MODEL_OUTPUT = DATA_DIR / "random_forest_customer_purchases.pkl"
TEST_PRED_OUTPUT = DATA_DIR / "test_predictions.csv"

# Carpeta donde se guardarán las gráficas
GRAPH_DIR = Path("C:/Users/tania/OneDrive/Documents/ML/ml25-electronenas/src/ml25/P01_customer_purchases/otros modelos/graficas")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Función para leer CSV
# -----------------------------
def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    fullfilename = os.path.abspath(file)
    return pd.read_csv(fullfilename)

# -----------------------------
# Función principal
# -----------------------------
def main():
    # -----------------------------
    # Leer datasets
    # -----------------------------
    train_df = read_csv("customer_purchases_train_final_preprocessed")
    test_df = read_csv("customer_purchases_test_final_preprocessed")

    # -----------------------------
    # Separar features y label
    # -----------------------------
    X_train_full = train_df.drop(columns=["label", "customer_id", "item_id"], errors="ignore")
    y_train_full = train_df["label"]

    # -----------------------------
    # Manejo de clases desbalanceadas
    # -----------------------------
    classes = np.array([0, 1])
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_full)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    # -----------------------------
    # División train/validation
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    # -----------------------------
    # Definir Random Forest + búsqueda de hiperparámetros
    # -----------------------------
    rf = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
    param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="f1",
        verbose=0,
        random_state=42,
        n_jobs=-1,
    )

    # -----------------------------
    # Entrenar modelo
    # -----------------------------
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    print("✅ Mejores hiperparámetros:", search.best_params_)

    # -----------------------------
    # Validación cruzada
    # -----------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_rf, X_train_full, y_train_full, cv=skf, scoring="f1_macro")
    print("\n🔁 Validación cruzada (F1 macro, 5 folds):")
    print("Scores individuales:", np.round(cv_scores, 4))
    print("Promedio F1 macro:", round(cv_scores.mean(), 4))

    # -----------------------------
    # Evaluación en validation set
    # -----------------------------
    y_val_pred = best_rf.predict(X_val)
    y_val_prob = best_rf.predict_proba(X_val)[:, 1]

    print("\n📈 Resultados en validation set:")
    print(classification_report(y_val, y_val_pred))
    print("ROC-AUC:", round(roc_auc_score(y_val, y_val_prob), 4))

    # -----------------------------
    # Matriz de confusión
    # -----------------------------
    cm = confusion_matrix(y_val, y_val_pred)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    print("\n📊 Matriz de confusión (validation set):")
    print(cm_df)

    # -----------------------------
    # Overfitting
    # -----------------------------
    train_score = best_rf.score(X_train, y_train)
    val_score = best_rf.score(X_val, y_val)
    diferencia = train_score - val_score
    print(f"\n🏋️ Score en train: {train_score:.4f}")
    print(f"🧪 Score en validation: {val_score:.4f}")
    print(f"📊 Diferencia train-validation: {diferencia:.4f}")
    if diferencia > 0.05:
        print("⚠️ Posible overfitting detectado.")
    else:
        print("✅ No se detecta overfitting significativo.")

    # -----------------------------
    # Guardar modelo
    # -----------------------------
    joblib.dump(best_rf, MODEL_OUTPUT)
    print(f"✅ Modelo Random Forest guardado en: {MODEL_OUTPUT}")

    # -----------------------------
    # Predicciones sobre test set (sin label)
    # -----------------------------
    X_test = test_df.drop(columns=["customer_id", "item_id"], errors="ignore")
    test_pred = best_rf.predict(X_test)
    test_prob = best_rf.predict_proba(X_test)[:, 1]

    test_results = test_df.copy()
    test_results["pred_label"] = test_pred
    test_results["pred_prob"] = test_prob
    test_results.to_csv(TEST_PRED_OUTPUT, index=False)
    print(f"💾 Predicciones guardadas en: {TEST_PRED_OUTPUT}")

    # ============================================================
    # 📊 VISUALIZACIONES DE RESULTADOS
    # ============================================================

    print("\n🎨 Generando gráficas en carpeta:", GRAPH_DIR)

    # --- 1. Matriz de confusión ---
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de confusión (Validation)")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "grafica_1_matriz_confusion.png", dpi=300)
    plt.close()

    # --- 2. Curva ROC ---
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    auc = roc_auc_score(y_val, y_val_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.2f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC")
    plt.legend()
    plt.savefig(GRAPH_DIR / "grafica_2_roc.png", dpi=300)
    plt.close()

    # --- 3. Precision-Recall ---
    precision, recall, _ = precision_recall_curve(y_val, y_val_prob)
    ap = average_precision_score(y_val, y_val_prob)
    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.savefig(GRAPH_DIR / "grafica_3_precision_recall.png", dpi=300)
    plt.close()

    # --- 4. Overfitting visual ---
    plt.figure(figsize=(4,4))
    plt.bar(["Entrenamiento", "Validación"], [train_score, val_score], color=["skyblue", "salmon"])
    plt.title("Comparación: Entrenamiento vs Validación")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.savefig(GRAPH_DIR / "grafica_4_overfitting.png", dpi=300)
    plt.close()

    # --- 5. Curvas de aprendizaje ---
    train_sizes, train_scores, val_scores = learning_curve(
        best_rf, X_train, y_train, cv=3, scoring="roc_auc",
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Train")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label="Validation")
    plt.title("Curva de aprendizaje")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.savefig(GRAPH_DIR / "grafica_5_curva_aprendizaje.png", dpi=300)
    plt.close()

    # --- 6. Importancia de características ---
    importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    plt.figure(figsize=(8,5))
    top10.plot(kind="barh")
    plt.title("Top 10 características más importantes")
    plt.xlabel("Importancia")
    plt.gca().invert_yaxis()
    plt.savefig(GRAPH_DIR / "grafica_6_importancia_variables.png", dpi=300)
    plt.close()

    print("✅ Todas las gráficas se generaron correctamente en:", GRAPH_DIR)


# ------------------------------------------------------------
# Ejecución del script
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
