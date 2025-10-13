import pandas as pd
import joblib
from pathlib import Path
import numpy as np
from datetime import datetime
from ml25.P01_customer_purchases.Procesamiento_de_datos import preprocess_customer_data, read_csv

CURRENT_FILE = Path(__file__).resolve()
RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODELS_DIR = CURRENT_FILE.parent / "trained_models"
DATA_DIR = CURRENT_FILE.parent / "../../datasets/customer_purchases/"

def load_model(model_name: str):
    filepath = MODELS_DIR / model_name
    print(f"Cargando modelo desde: {filepath}")
    model = joblib.load(filepath)
    return model

def run_inference(model_name: str, test_filename: str):
    df_test = read_csv(test_filename)
    X_test = preprocess_customer_data(df_test)

# a estas lineas se les quita el # cuando ya este listo el modelo
    # model = load_model(model_name)
    # preds = model.predict(X_test)
    # probs = model.predict_proba(X_test)[:, 1]

    np.random.seed(42)
    preds = np.random.choice([0, 1], size=len(X_test))
    probs = np.random.rand(len(X_test))

    results = pd.DataFrame({
        "ID": X_test.index,
        "prediction": preds,
        "probability": probs
    })

    output_path = RESULTS_DIR / f"{test_filename}_predictions.csv"
    results.to_csv(output_path, index=False)
    print(f"Resultados guardados en: {output_path}")
    return results

def plot_roc(y_true, y_proba):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    model_name = "modelo_final.pkl"
    test_filename = "customer_purchases_test"
    run_inference(model_name, test_filename)