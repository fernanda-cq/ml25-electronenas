# ==========================
# graficas_controladas.py
# ==========================
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8,5)
REF_DATE = datetime(2025, 9, 21)

# ---- QUÉ GRÁFICAS MOSTRAR ----
PLOTS = {
    "p1": True,   # 1) Distribución # compras por cliente
    "p2": True,   # 2) Años desde release vs # compras (mejorada)
    "p3": True,   # 3) Distribución de precios
    "p4": True,   # 4) Edad vs total gastado
    "p5": True,   # 5) Género vs categoría de prenda
    "p6": True,   # 6) Rating promedio vs Precio
    "p9": False,  # 9) Total de compras por dispositivo  << DESACTIVADA
}

# ---- Paths / carga ----
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent / "datasets" / "customer_purchases" / "customer_purchases_train.csv"
df = pd.read_csv(CSV_PATH)

# ---- Prepro ----
for col in ["customer_date_of_birth", "customer_signup_date", "item_release_date"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

df["customer_age"] = (REF_DATE - df["customer_date_of_birth"]).dt.days // 365
df["customer_seniority"] = (REF_DATE - df["customer_signup_date"]).dt.days // 365
if "item_release_date" in df.columns:
    df["item_years_since_release"] = (REF_DATE - df["item_release_date"]).dt.days / 365.0

# ==============================
# 1) Distribución compras/cliente
# ==============================
if PLOTS["p1"]:
    compras_por_cliente = df.groupby('customer_id')['item_id'].count()
    plt.hist(compras_por_cliente.values, bins=30, edgecolor='black')
    plt.title("Distribución de # de compras por cliente")
    plt.xlabel("# de compras por cliente"); plt.ylabel("# de clientes")
    plt.show()

# ==============================================
# 2) Años desde release vs # de compras (MEJORADA)
# ==============================================
if PLOTS["p2"] and "item_years_since_release" in df.columns:
    yrs = df["item_years_since_release"].dropna().round().astype(int)
    rel_counts = yrs.value_counts().sort_index()
    plt.bar(rel_counts.index, rel_counts.values, edgecolor="black")
    if len(rel_counts) >= 3:
        roll = rel_counts.sort_index().rolling(3, center=True).mean()
        plt.plot(roll.index, roll.values, marker="o")
    plt.title("Años desde release vs # de compras")
    plt.xlabel("Años desde release (redondeado)"); plt.ylabel("# Compras")
    plt.show()

# ==============================
# 3) Distribución de precios
# ==============================
if PLOTS["p3"]:
    plt.hist(df['item_price'].dropna(), bins=30, edgecolor='black')
    plt.title("Distribución de precios")
    plt.xlabel("Precio"); plt.ylabel("Frecuencia")
    plt.show()

# ==============================
# 4) Edad vs total gastado
# ==============================
if PLOTS["p4"]:
    gasto_por_edad = df.groupby('customer_age')['item_price'].sum().dropna()
    plt.scatter(gasto_por_edad.index, gasto_por_edad.values, alpha=0.7)
    plt.title("Edad vs Total gastado")
    plt.xlabel("Edad (años)"); plt.ylabel("Total gastado")
    plt.show()

# ==============================
# 5) Género vs Categoría de prenda
# ==============================
if PLOTS["p5"]:
    genero_categoria = df.groupby(['item_category','customer_gender']).size().unstack(fill_value=0)
    genero_categoria.plot(kind='bar', stacked=True, figsize=(10,5))
    plt.title("Género vs Categoría de prenda")
    plt.xlabel("Categoría"); plt.ylabel("# Compras")
    plt.show()

# ==============================
# 6) Rating promedio vs Precio
# ==============================
if PLOTS["p6"]:
    plt.scatter(df['item_avg_rating'], df['item_price'], alpha=0.5)
    plt.title("Rating promedio vs Precio del producto")
    plt.xlabel("Rating promedio"); plt.ylabel("Precio del producto")
    plt.show()

# ==============================
# 9) Total de compras por dispositivo
# ==============================
if PLOTS["p9"] and "purchase_device" in df.columns:
    dev_counts = df['purchase_device'].value_counts()
    dev_counts.plot(kind='bar', edgecolor='black')
    plt.title("Total de compras por dispositivo")
    plt.xlabel("Dispositivo"); plt.ylabel("# Compras")
    plt.show()
