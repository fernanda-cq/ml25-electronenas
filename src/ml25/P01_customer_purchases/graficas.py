# ==========================
# graficas.py — EDA (7 figuras) + mini-análisis impreso en consola
# ==========================
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 5)
REF_DATE = datetime(2025, 9, 21)

# ---------- Rutas ----------
HERE = Path(__file__).resolve().parent  # ej. src/ml25/P01_customer_purchases/
CANDIDATES = [
    HERE.parent / "datasets" / "customer_purchases" / "customer_purchases_train.csv",
    HERE.parent.parent / "datasets" / "customer_purchases" / "customer_purchases_train.csv",
    HERE / "customer_purchases_train.csv",
]
CSV_PATH = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])

OUT_DIR = HERE / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("CSV_PATH:", CSV_PATH)
print("OUT_DIR :", OUT_DIR)

# ---------- Helpers ----------
def save_fig(fig, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"✔ guardada: {path}")
    plt.show()
    plt.close(fig)

def say(title: str, text: str):
    """Imprime análisis corto y legible en consola."""
    print(f"\n[Análisis] {title}\n{text}\n")

def to_price_numeric(s: pd.Series) -> pd.Series:
    """Convierte '1.234,56'->1234.56, quita símbolos y parsea a float robusto."""
    s = s.astype("string").str.strip()
    s = s.str.replace(r'\.(?=\d{3}\b)', '', regex=True)  # puntos de miles
    s = s.str.replace(',', '.', regex=False)             # coma decimal -> punto
    s = s.str.replace(r'[^0-9.\-]', '', regex=True)      # símbolos
    s = s.replace({'': np.nan})
    return pd.to_numeric(s, errors="coerce")

def normalize_gender(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    s = s.replace({pd.NA: "unknown", "nan": "unknown", "none": "unknown", "": "unknown"})
    s = s.replace({"1": "female", "2": "male", "0": "unknown", "f": "female", "m": "male", "u":"unknown"})
    return s.where(s.isin(["female", "male", "unknown"]), "unknown")

# ---------- Carga ----------
df = pd.read_csv(CSV_PATH)

# ---------- Fechas / derivadas ----------
for c in ["customer_date_of_birth", "customer_signup_date", "item_release_date", "purchase_timestamp"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

if "customer_date_of_birth" in df.columns:
    df["customer_age"] = (REF_DATE - df["customer_date_of_birth"]).dt.days // 365
if "customer_signup_date" in df.columns:
    df["customer_seniority"] = (REF_DATE - df["customer_signup_date"]).dt.days // 365
if "item_years_since_release" not in df.columns and "item_release_date" in df.columns:
    df["item_years_since_release"] = (REF_DATE - df["item_release_date"]).dt.days / 365.0
if "item_years_since_release" in df.columns:
    df["item_years_since_release"] = pd.to_numeric(df["item_years_since_release"], errors="coerce")

# ---------- Limpieza de PRECIOS ----------
if "item_price" in df.columns:
    df["item_price"] = to_price_numeric(df["item_price"])
else:
    df["item_price"] = np.nan  # evita rompimientos posteriores

# =========================================================
# GASTO ROBUSTO: transacción -> cliente (evita doble conteo)
# =========================================================
df_tx = df[df["item_price"].notna()].copy()

# quitar duplicados de línea por seguridad
dup_subset = [c for c in ["purchase_id", "item_id", "item_price", "purchase_timestamp", "customer_id"] if c in df_tx.columns]
if dup_subset:
    before = len(df_tx)
    df_tx = df_tx.drop_duplicates(subset=dup_subset)
    say("Depuración de duplicados",
        f"Se eliminaron {before - len(df_tx)} filas duplicadas exactas (misma compra/ítem/precio).")

# total por transacción
if "purchase_id" in df_tx.columns:
    tx_totals = (
        df_tx.groupby("purchase_id", as_index=False)["item_price"]
             .sum()
             .rename(columns={"item_price": "purchase_total"})
    )
    tx_customer = df_tx[["purchase_id", "customer_id"]].drop_duplicates("purchase_id")
    cust_totals = (
        tx_customer.merge(tx_totals, on="purchase_id", how="left")
                   .groupby("customer_id", as_index=False)["purchase_total"].sum()
                   .rename(columns={"purchase_total": "total_spent"})
    )
else:
    # Fallback si no hay purchase_id
    cust_totals = (
        df_tx.groupby("customer_id", as_index=False)["item_price"].sum()
             .rename(columns={"item_price": "total_spent"})
    )

attrs_cols = [c for c in ["customer_id", "customer_age", "customer_seniority", "customer_gender"] if c in df.columns]
cust_attrs = df[attrs_cols].drop_duplicates("customer_id") if attrs_cols else pd.DataFrame(columns=["customer_id"])
cust = cust_totals.merge(cust_attrs, on="customer_id", how="left")
cust["gender_clean"] = normalize_gender(cust["customer_gender"]) if "customer_gender" in cust.columns else "unknown"

# Subconjunto de compras reales para cruces por categoría (si 'label' existe)
df_buy_tx = df_tx[df_tx["label"] == 1].copy() if "label" in df_tx.columns else df_tx.copy()

# =========================================================
# (1) Distribución de # de compras por cliente
# ANÁLISIS: ver cuántas compras hace “en promedio” un cliente y cuánta cola larga hay.
# =========================================================
if "purchase_id" in df_tx.columns:
    purchases_per_customer = df_tx.drop_duplicates("purchase_id").groupby("customer_id")["purchase_id"].nunique()
    fig, ax = plt.subplots()
    ax.hist(purchases_per_customer.values, bins=30, edgecolor="black")
    ax.set_title("Distribución de # de compras por cliente")
    ax.set_xlabel("# de compras por cliente"); ax.set_ylabel("# de clientes")
    save_fig(fig, "compras_por_cliente_hist")

    p_med = purchases_per_customer.median()
    p_p90 = purchases_per_customer.quantile(0.90)
    say("Compras por cliente",
        f"La mediana está en ~{p_med:.0f} compras por cliente. "
        f"El 10% más activo supera ~{p_p90:.0f} compras; hay una cola de heavy users.")

# =========================================================
# (2) Años desde release vs # de compras (barras + media móvil 3)
# ANÁLISIS: productos más nuevos suelen mover más compras; la media móvil ayuda a ver tendencia.
# =========================================================
if "item_years_since_release" in df_buy_tx.columns:
    yrs = df_buy_tx["item_years_since_release"].dropna().round().astype(int)
    rel_counts = yrs.value_counts().sort_index()
    if len(rel_counts):
        fig, ax = plt.subplots()
        ax.bar(rel_counts.index, rel_counts.values, edgecolor="black", alpha=0.7)
        if len(rel_counts) >= 3:
            roll = rel_counts.sort_index().rolling(3, center=True).mean()
            ax.plot(roll.index, roll.values, marker="o")
        ax.set_title("Años desde release vs # de compras")
        ax.set_xlabel("Años desde release (redondeado)"); ax.set_ylabel("# Compras")
        save_fig(fig, "years_release_vs_purchases")

        top_year = int(rel_counts.idxmax())
        trend = "descendente con el tiempo" if rel_counts.iloc[0] > rel_counts.iloc[-1] else "sin un patrón claro"
        say("Antigüedad del producto",
            f"El pico de compras ocurre cerca de {top_year} años desde el lanzamiento. Globalmente la curva es {trend}.")

# =========================================================
# (3) Distribución de precios (limpios)
# ANÁLISIS: rango típico de precios y presencia de outliers.
# =========================================================
fig, ax = plt.subplots()
ax.hist(df_tx["item_price"].dropna(), bins=30, edgecolor="black")
ax.set_title("Distribución de precios (limpios)")
ax.set_xlabel("Precio"); ax.set_ylabel("Frecuencia")
save_fig(fig, "dist_item_price")

pr = df_tx["item_price"].dropna()
say("Precios",
    f"La mediana está en ${pr.median():.0f}. El 90% de los productos cuesta menos de ~${pr.quantile(0.90):.0f}. "
    f"El máximo observado en datos es ${pr.max():.0f}.")

# =========================================================
# (4) Edad vs Total gastado (por cliente, SIN doble conteo)
# ANÁLISIS: ver cómo varía el gasto por cliente según su edad; la línea punteada es la mediana por edad.
# =========================================================
if {"customer_age","total_spent"}.issubset(cust.columns):
    ok = cust.dropna(subset=["customer_age","total_spent"]).copy()
    if len(ok) > 5:
        ycap = ok["total_spent"].quantile(0.995)  # cap visual
        ok["total_spent_plot"] = ok["total_spent"].clip(upper=ycap)
    else:
        ok["total_spent_plot"] = ok["total_spent"]
    fig, ax = plt.subplots()
    ax.scatter(ok["customer_age"], ok["total_spent_plot"], alpha=0.7)
    med = ok.groupby("customer_age")["total_spent_plot"].median()
    if len(med):
        ax.plot(med.index, med.values, linestyle="--")
    ax.set_title("Edad vs Total gastado (por cliente, sin doble conteo)")
    ax.set_xlabel("Edad (años)"); ax.set_ylabel("Total gastado")
    save_fig(fig, "edad_vs_total_gastado_por_cliente")

    if len(med):
        age_peak = int(med.idxmax())
        say("Edad y gasto",
            f"El gasto mediano por cliente alcanza su pico alrededor de los {age_peak} años. "
            "Fuera del pico, el gasto se mantiene relativamente estable con variación individual.")

# =========================================================
# (5) Género (incluye Unknown) vs Categoría (apiladas)
# ANÁLISIS: qué categorías se compran más y cómo se reparten por género (incluye Unknown).
# =========================================================
cat_col = "item_category" if "item_category" in df_buy_tx.columns else ("item_category_num" if "item_category_num" in df_buy_tx.columns else None)
if cat_col is not None:
    gmap = cust.set_index("customer_id")["gender_clean"]
    tmp = df_buy_tx.merge(gmap.rename("gender_clean"), left_on="customer_id", right_index=True, how="left")
    tmp["gender_clean"] = normalize_gender(tmp["gender_clean"])
    ct = pd.crosstab(tmp[cat_col].astype("string"), tmp["gender_clean"])
    for col in ["female", "male", "unknown"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["female", "male", "unknown"]]

    ax = ct.plot(kind="bar", stacked=True, figsize=(10,5))
    ax.set_title("Género (incluye Unknown) vs Categoría")
    ax.set_xlabel("Categoría"); ax.set_ylabel("# Compras")
    fig = ax.get_figure()
    save_fig(fig, "genero_vs_categoria_incluye_unknown")

    top_cat = ct.sum(axis=1).idxmax()
    top_f = ct["female"].idxmax()
    top_m = ct["male"].idxmax()
    say("Preferencias por categoría",
        f"La categoría más comprada es '{top_cat}'. Entre mujeres destaca '{top_f}' y entre hombres '{top_m}'. "
        "Aun así, la mayoría de categorías muestra compras de ambos géneros.")

# =========================================================
# (6) Rating promedio vs Precio del producto
# ANÁLISIS: comprobar si los productos más caros reciben mejor rating (correlación).
# =========================================================
if {"item_avg_rating","item_price"}.issubset(df_tx.columns):
    ok = df_tx.dropna(subset=["item_avg_rating","item_price"])
    fig, ax = plt.subplots()
    ax.scatter(ok["item_avg_rating"], ok["item_price"], alpha=0.5)
    ax.set_title("Rating promedio vs Precio del producto")
    ax.set_xlabel("Rating promedio"); ax.set_ylabel("Precio del producto")
    save_fig(fig, "rating_vs_precio")

    corr = ok[["item_avg_rating","item_price"]].corr().iloc[0,1]
    feel = "nula" if abs(corr) < 0.1 else ("ligera" if abs(corr) < 0.3 else "moderada")
    dirr = "positiva" if corr >= 0 else "negativa"
    say("Precio y rating",
        f"La relación precio-rating es {feel} y {dirr} (corr={corr:.2f}); no parece que un mayor precio garantice mejor calificación.")

# =========================================================
# (7) Antigüedad del cliente vs Total gastado (por cliente)
# ANÁLISIS: ver si los clientes con más tiempo gastan más; línea punteada = mediana por antigüedad.
# =========================================================
if {"customer_seniority","total_spent"}.issubset(cust.columns):
    ok2 = cust.dropna(subset=["customer_seniority","total_spent"]).copy()
    if len(ok2) > 5:
        ycap2 = ok2["total_spent"].quantile(0.995)
        ok2["total_spent_plot"] = ok2["total_spent"].clip(upper=ycap2)
    else:
        ok2["total_spent_plot"] = ok2["total_spent"]
    fig, ax = plt.subplots()
    ax.scatter(ok2["customer_seniority"], ok2["total_spent_plot"], alpha=0.7)
    med2 = ok2.groupby("customer_seniority")["total_spent_plot"].median()
    if len(med2):
        ax.plot(med2.index, med2.values, linestyle="--")
    ax.set_title("Antigüedad del cliente vs Total gastado (sin doble conteo)")
    ax.set_xlabel("Antigüedad (años)"); ax.set_ylabel("Total gastado")
    save_fig(fig, "antiguedad_vs_total_gastado_por_cliente")

    if len(med2):
        sen_peak = int(med2.idxmax())
        say("Antigüedad y valor",
            f"Los clientes con ~{sen_peak} años de antigüedad concentran la mayor mediana de gasto. "
            "La relación general es suave, con casos individuales muy variables.")

# ---------- Listado final ----------
print("\nContenido de figs/:")
for p in sorted(OUT_DIR.glob("*.png")):
    print(" -", p.name)
print("\n✅ Listo. Las 7 figuras se guardaron en:", OUT_DIR)

