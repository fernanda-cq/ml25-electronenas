# model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
from pathlib import Path
import os

DATA_DIR = Path(__file__).resolve().parent.parent / "datasets/customer_purchases/"
MODEL_OUTPUT = DATA_DIR / "random_forest_customer_purchases.pkl"

def create_model(class_weight_dict=None, **rf_kwargs):
    """
    Crea el modelo Random Forest con búsqueda de hiperparámetros
    
    Args:
        class_weight_dict: Diccionario de pesos de clases
        **rf_kwargs: Argumentos adicionales para RandomForestClassifier
    """
    # Configuración base + argumentos personalizados
    base_config = {
        'random_state': 42,
        'class_weight': class_weight_dict
    }
    base_config.update(rf_kwargs)
    
    rf = RandomForestClassifier(**base_config)
    
    param_dist = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    
    return rf, param_dist

def create_search_cv(estimator, param_dist, n_iter=20, cv=3, scoring="f1", **kwargs):
    """
    Crea RandomizedSearchCV con configuración flexible
    """
    base_config = {
        'estimator': estimator,
        'param_distributions': param_dist,
        'n_iter': n_iter,
        'cv': cv,
        'scoring': scoring,
        'verbose': 1,
        'random_state': 42,
        'n_jobs': -1,
    }
    base_config.update(kwargs)
    
    return RandomizedSearchCV(**base_config)

def save_model(model):
    """
    Guarda el modelo entrenado
    """
    try:
        # Crear directorio si no existe
        MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT)
        print(f"✅ Modelo guardado en {MODEL_OUTPUT}")
        return True
    except Exception as e:
        print(f"❌ Error guardando modelo: {e}")
        return False

def load_model():
    """
    Carga un modelo guardado
    """
    try:
        if not MODEL_OUTPUT.exists():
            raise FileNotFoundError(f"Modelo no encontrado en {MODEL_OUTPUT}")
        
        model = joblib.load(MODEL_OUTPUT)
        print(f"✅ Modelo cargado desde {MODEL_OUTPUT}")
        return model
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None

def get_model_info():
    """
    Obtiene información del modelo guardado
    """
    if MODEL_OUTPUT.exists():
        model = load_model()
        if hasattr(model, 'best_params_'):
            return {
                'best_params': model.best_params_,
                'best_score': model.best_score_,
                'model_type': type(model.estimator).__name__
            }
    return None