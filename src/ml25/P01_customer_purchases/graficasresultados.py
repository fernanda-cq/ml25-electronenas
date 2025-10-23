#!/usr/bin/env python3
"""
generate_training_plots_from_training.py

Script independiente que lee:
 - datasets/customer_purchases/customer_purchases_train_with_ids.csv
 - datasets/customer_purchases/customer_purchases_test_with_ids.csv
 - datasets/customer_purchases/random_forest_customer_purchases.pkl

Reconstruye preprocesado / undersampling / split (mismos seeds),
ALINEA columnas al modelo guardado (orden + faltantes/extra),
y genera gráficas de:
 - ROC (validation)
 - Precision-Recall (validation)
 - Confusion Matrix (raw y normalizada) (validation)
 - Feature importances (top N)
 - Boxplot de CV (recalcula cross-val sobre el conjunto balanceado)

Guarda las imágenes en datasets/customer_purchases/plots_training/.
No modifica ningun archivo existente.
"""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools

# -----------------------------
# Rutas (NO modifican archivos)
# -----------------------------
HERE = Path(__file__).resolve()
DATA_DIR = (HERE / "../../datasets/customer_purchases/").resolve()
MODEL_FILE = DATA_DIR / "random_forest_customer_purchases.pkl"
TRAIN_CSV = DATA_DIR / "customer_purchases_train_with_ids.csv"
TEST_CSV = DATA_DIR / "customer_purchases_test_with_ids.csv"
PLOTS_DIR = DATA_DIR / "plots_training"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Lista stop-words (igual que en tu training)
# -----------------------------
features_stop_words = [
    'item_title_bow_the', 'item_title_bow_you', 'item_title_bow_your',
    'item_title_bow_that', 'item_title_bow_for', 'item_title_bow_with',
    'item_title_bow_have', 'item_title_bow_must', 'item_title_bow_need',
    'item_title_bow_every', 'item_title_bow_out', 'item_title_bow_up',
    'item_title_bow_occasion', 'item_title_bow_stand', 'item_title_bow_step',
    'item_title_bow_style'
]

# -----------------------------
# Funciones de plotting
# -----------------------------
def save_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1],'--', linewidth=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (validation)'); plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path); plt.close()

def save_pr(y_true, y_score, path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(7,6))
    plt.plot(recall, precision, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (validation)'); plt.legend(loc='upper right'); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path); plt.close()

def save_cm(y_true, y_pred, path, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(invalid='ignore', divide='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', aspect='auto')
    plt.title('Confusion Matrix (validation)' + (' - normalized' if normalize else ''))
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.colorbar()
    ticks = np.arange(cm.shape[0]); plt.xticks(ticks); plt.yticks(ticks)
    fmt = '.2f' if normalize else 'd'; thresh = cm.max()/2 if cm.size else 0.5
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt), ha='center', va='center',
                 color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout(); plt.savefig(path); plt.close()

def save_feature_importances(features, importances, path, top_n=30):
    df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False).head(top_n)
    plt.figure(figsize=(8, max(4, 0.25*len(df))))
    plt.barh(range(len(df)), df['importance'].values[::-1])
    plt.yticks(range(len(df)), df['feature'].values[::-1])
    plt.xlabel('Importance'); plt.title(f'Top {len(df)} Feature Importances')
    plt.tight_layout(); plt.savefig(path); plt.close()

def save_cv_boxplot(cv_scores, path):
    plt.figure(figsize=(6,4))
    plt.boxplot(cv_scores, vert=True)
    plt.title('Cross-val ROC-AUC (5-fold)')
    plt.ylabel('ROC-AUC'); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path); plt.close()

# -----------------------------
# Validaciones iniciales (archivos existen)
# -----------------------------
def must_exist(p: Path, name: str):
    if not p.exists():
        print(f"ERROR: {name} no encontrado en: {p}")
        sys.exit(1)

must_exist(TRAIN_CSV, "TRAIN CSV")
must_exist(TEST_CSV, "TEST CSV")
must_exist(MODEL_FILE, "MODELO")

# -----------------------------
# Cargar datos y modelo
# -----------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
model = joblib.load(MODEL_FILE)
print("✅ Datos y modelo cargados (solo lectura).")

# -----------------------------
# Función para alinear columnas al modelo guardado
# -----------------------------
def align_features_to_model(X: pd.DataFrame, model):
    """
    Reordena X según model.feature_names_in_ (si existe).
    Añade columnas faltantes con ceros y elimina columnas extra.
    Devuelve X_aligned (copiado).
    """
    X = X.copy()
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        # fallback: si no existe feature_names_in_, intentar inferir usando el índice de entrenamiento guardado
        # pero si no hay forma segura, devolvemos X tal cual (riesgo de error posterior)
        print("⚠️ WARNING: el modelo no tiene 'feature_names_in_'. Se conservará el orden actual de X (podría fallar).")
        return X

    X_cols = list(X.columns)
    expected_set = set(expected)
    X_set = set(X_cols)

    missing = [c for c in expected if c not in X_set]
    extra = [c for c in X_cols if c not in expected_set]

    if missing:
        print(f"⚠️ Columnas faltantes detectadas ({len(missing)}). Se crearán con ceros (ej: {missing[:10]}{'...' if len(missing)>10 else ''}).")
        for c in missing:
            X[c] = 0

    if extra:
        print(f"⚠️ Columnas extra detectadas en X ({len(extra)}). Se eliminarán (ej: {extra[:10]}{'...' if len(extra)>10 else ''}).")
        X = X.drop(columns=extra)

    # Reordenar en el orden exacto esperado
    try:
        X_aligned = X[expected].copy()
    except KeyError as e:
        print("ERROR: no se pudo reordenar columnas exactamente como las espera el modelo. Detalle:", e)
        raise

    # Comprobación final
    if list(X_aligned.columns) != expected:
        print("ERROR: fallo al alinear columnas exactamente en el mismo orden que el modelo.")
        raise RuntimeError("No se pudo alinear columnas con feature_names_in_ del modelo.")
    else:
        print(f"✅ Alineado exitoso con el modelo: {len(expected)} columnas (faltaron: {len(missing)}, extras eliminadas: {len(extra)})")

    return X_aligned

# -----------------------------
# Reconstruir preprocesado y undersampling (igual al training)
# -----------------------------
y_full = train_df["label"].copy()
train_feats = train_df.drop(columns=["label", "customer_id"], errors="ignore")
test_feats = test_df.drop(columns=["customer_id"], errors="ignore")

common_cols = list(set(train_feats.columns) & set(test_feats.columns))
stop_existing = [f for f in features_stop_words if f in common_cols]
common_filtered = [c for c in common_cols if c not in stop_existing]

X_train_full = train_feats[common_filtered]
X_test = test_feats[common_filtered]

train_data = pd.concat([X_train_full, y_full], axis=1)
class_0 = train_data[train_data['label'] == 0]
class_1 = train_data[train_data['label'] == 1]
if len(class_0) == 0 or len(class_1) == 0:
    print("ERROR: una de las clases está vacía. Abortando.")
    sys.exit(1)

undersample_ratio = 2
n_samples_class_1 = len(class_0) * undersample_ratio
class_1_undersampled = class_1.sample(n=min(n_samples_class_1, len(class_1)), random_state=42)
balanced_data = pd.concat([class_0, class_1_undersampled], axis=0)
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

X_bal = balanced_data.drop(columns=['label'])
y_bal = balanced_data['label']

# Asegurar columnas coincidentes con test (igual que en training)
final_common_cols = list(set(X_bal.columns).intersection(set(X_test.columns)))
if len(final_common_cols) == 0:
    print("ERROR: no hay columnas en común entre train y test tras filtrado. Abortando.")
    sys.exit(1)

X_bal = X_bal[final_common_cols]
X_test = X_test[final_common_cols]

# Reproducir split train/val (80/20, stratify, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
print(f"✅ Reconstruido split -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# -----------------------------
# Alinear dataframes al modelo guardado (orden + faltantes/extras)
# -----------------------------
print("🔧 Alineando features de los datasets al orden que el modelo espera...")
X_train = align_features_to_model(X_train, model)
X_val   = align_features_to_model(X_val, model)
X_bal   = align_features_to_model(X_bal, model)
X_test  = align_features_to_model(X_test, model)

# -----------------------------
# Obtener probabilidades predichas sobre validation (usando X_val alineado)
# -----------------------------
print("🔎 Calculando probabilidades/predicciones sobre validation...")
if hasattr(model, "predict_proba"):
    try:
        y_val_prob = model.predict_proba(X_val)[:, 1]
    except Exception as e:
        print("ERROR al usar predict_proba:", e)
        # Si predict_proba falla (raro si se alineó), intentar fallback a predict
        y_val_prob = model.predict(X_val)
else:
    try:
        y_val_prob = model.decision_function(X_val)
    except Exception:
        y_val_prob = model.predict(X_val)

# -----------------------------
# Generar y guardar plots
# -----------------------------
print("🔖 Generando gráficas en:", PLOTS_DIR)

# ROC y PR
save_roc(y_val, y_val_prob, PLOTS_DIR / "roc_validation.png")
save_pr(y_val, y_val_prob, PLOTS_DIR / "pr_validation.png")

# Confusion matrices
y_val_pred = model.predict(X_val)
save_cm(y_val, y_val_pred, PLOTS_DIR / "cm_validation.png", normalize=False)
save_cm(y_val, y_val_pred, PLOTS_DIR / "cm_validation_norm.png", normalize=True)

# Feature importances
if hasattr(model, "feature_importances_"):
    # usamos las columnas alineadas de X_bal para saber el orden exacto
    save_feature_importances(X_bal.columns, model.feature_importances_, PLOTS_DIR / "feature_importances.png", top_n=30)
else:
    print("ℹ️ Modelo no tiene attribute 'feature_importances_'")

# Cross-val recalculado sobre X_bal (puede tardar)
try:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_bal, y_bal, cv=skf, scoring="roc_auc", n_jobs=-1)
    save_cv_boxplot(cv_scores, PLOTS_DIR / "cv_boxplot.png")
    print("✅ cross-val ROC-AUC:", np.round(cv_scores, 4), "mean:", cv_scores.mean())
except Exception as e:
    print("⚠️ Warning: cross_val_score falló:", e)

# Listar archivos creados
print("Gráficas generadas en:", PLOTS_DIR)
for f in sorted(PLOTS_DIR.iterdir()):
    print(" -", f.name)

print("🎉 Listo. Este script no modifica tu código base ni sus archivos.")
