import sys
from pathlib import Path

# Agregar el directorio actual al path para importar data_processing.py
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import joblib
from data_processing import read_test_data


# -----------------------------
# Configuración de rutas
# -----------------------------
CURRENT_FILE = Path(__file__).resolve()
RESULTS_DIR = CURRENT_FILE.parent / "test_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MODEL_PATH = Path(
    "C:/Users/tania/OneDrive/Documents/ML/ml25-electronenas/src/ml25/datasets/customer_purchases/random_forest_customer_purchases.pkl"
)

# -----------------------------
# Función de inferencia
# -----------------------------
def run_inference(model_path: Path, X: pd.DataFrame) -> pd.DataFrame:
    print(f"🔍 Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    results = pd.DataFrame({
        "prediction": preds,
        "probability": probs
    })
    return results

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # 1️⃣ Cargar datos procesados
    X = read_test_data()
    print(f"✅ Datos de test cargados: {X.shape}")

    # 2️⃣ Cargar CSV original de test para obtener los IDs
    test_path = Path(
        "C:/Users/tania/OneDrive/Documents/ML/ml25-electronenas/src/ml25/datasets/customer_purchases/customer_purchases_test.csv"
    )
    test_original = pd.read_csv(test_path)
    customer_ids = test_original["customer_id"]

    # 3️⃣ Quitar columnas que el modelo no usó
    X_infer = X.drop(columns=["customer_id", "item_id"], errors="ignore")

    # 4️⃣ Ejecutar inferencia
    results = run_inference(MODEL_PATH, X_infer)

    # 5️⃣ Combinar IDs con predicciones
    submission = pd.DataFrame({
        "ID": test_original["purchase_id"],
        "pred": results["prediction"]
    })

    # 6️⃣ Guardar archivo final sin NaN
    submission = submission.fillna(0)  # reemplaza posibles vacíos por 0
    submission_file = RESULTS_DIR / "submission.csv"
    submission.to_csv(submission_file, index=False)

    print(f"🚀 Archivo para Kaggle generado: {submission_file}")
    print(submission.head())
