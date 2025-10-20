from datetime import datetime
import os
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import numpy as np

# -----------------------------
# Paths
# -----------------------------
DATA_COLLECTED_AT = datetime(2025, 9, 21).date()
CURRENT_FILE = Path(__file__).resolve()
DATA_DIR = Path("C:/Users/tania/OneDrive/Documents/ML/ml25-electronenas/src/ml25/datasets/customer_purchases")

MODEL_OUTPUT = DATA_DIR / "decision_tree_customer_purchases.pkl"
TEST_PRED_OUTPUT = DATA_DIR / "test_predictions_decision_tree.csv"

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
    # Leer datasets separados
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
    # Definir Decision Tree y búsqueda de hiperparámetros
    # -----------------------------
    dt = DecisionTreeClassifier(random_state=42, class_weight=class_weight_dict)
    param_dist = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "criterion": ["gini", "entropy"]
    }

    search = RandomizedSearchCV(
        estimator=dt,
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
    best_dt = search.best_estimator_
    print("✅ Mejores hiperparámetros:", search.best_params_)

    # -----------------------------
    # Validación cruzada F1 macro
    # -----------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_dt, X_train_full, y_train_full, cv=skf, scoring="f1_macro")
    print("\n🔁 Validación cruzada (F1 macro, 5 folds):")
    print("Scores individuales:", np.round(cv_scores, 4))
    print("Promedio F1 macro:", round(cv_scores.mean(), 4))

    # -----------------------------
    # Evaluación en validation set
    # -----------------------------
    y_val_pred = best_dt.predict(X_val)
    y_val_prob = best_dt.predict_proba(X_val)[:, 1]

    print("\n📈 Resultados en validation set:")
    print(classification_report(y_val, y_val_pred))
    print("ROC-AUC:", round(roc_auc_score(y_val, y_val_prob), 4))

    # -----------------------------
    # Matriz de confusión
    # -----------------------------
    cm = confusion_matrix(y_val, y_val_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"]
    )
    print("\n📊 Matriz de confusión (validation set):")
    print(cm_df)

    # -----------------------------
    # Overfitting
    # -----------------------------
    train_score = best_dt.score(X_train, y_train)
    val_score = best_dt.score(X_val, y_val)
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
    joblib.dump(best_dt, MODEL_OUTPUT)
    print(f"✅ Modelo Decision Tree guardado en: {MODEL_OUTPUT}")

    # -----------------------------
    # Predicciones sobre test set
    # -----------------------------
    X_test = test_df.drop(columns=["customer_id", "item_id"], errors="ignore")
    test_pred = best_dt.predict(X_test)
    test_prob = best_dt.predict_proba(X_test)[:, 1]

    test_results = test_df.copy()
    test_results["pred_label"] = test_pred
    test_results["pred_prob"] = test_prob
    test_results.to_csv(TEST_PRED_OUTPUT, index=False)
    print(f"✅ Predicciones guardadas en: {TEST_PRED_OUTPUT}")

if __name__ == "__main__":
    main()