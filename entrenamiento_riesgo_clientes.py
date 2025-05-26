import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report

# =============================
# 📥 CARGA Y FILTRADO
# =============================
df = pd.read_excel("demanda_limpia.xlsx")
df.columns = df.columns.str.strip().str.lower()
df = df[df["cantidad"] > 0]
df["cliente_id"] = df["cliente_id"].astype(str)

# =============================
# 🧠 FEATURES POR CLIENTE
# =============================
df_group = df.groupby("cliente_id").agg({
    "monto": ["sum", "mean", "std"],
    "cantidad": ["sum", "mean", "std"]
})
df_group.columns = ["monto_total", "monto_promedio", "monto_std",
                    "cantidad_total", "cantidad_promedio", "cantidad_std"]
df_group.reset_index(inplace=True)
df_group.fillna(0, inplace=True)
df_group["monto_unitario_prom"] = df_group["monto_total"] / df_group["cantidad_total"]

# =============================
# 🔢 CLUSTERING (n=2 con verificación)
# =============================
X_cluster = df_group.drop(columns=["cliente_id"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print("🔁 Ejecutando clustering (n_clusters=2)...")
for intento in range(5):
    kmeans = KMeans(n_clusters=2, random_state=42 + intento, n_init=10)
    df_group["riesgo"] = kmeans.fit_predict(X_scaled)
    conteo = df_group["riesgo"].value_counts()
    if conteo.min() >= 5:
        print(f"✅ Clústeres válidos encontrados (intento {intento+1}):\n{conteo}")
        break
else:
    print("❌ No se lograron clústeres equilibrados. Abortando...")
    exit()

# Ordenar clústeres por monto_total
centros = pd.DataFrame(kmeans.cluster_centers_, columns=X_cluster.columns)
orden = centros["monto_total"].sort_values().index
mapa_riesgo = {old: new for new, old in enumerate(orden)}
df_group["riesgo"] = df_group["riesgo"].map(mapa_riesgo)

# =============================
# 🤖 ENTRENAMIENTO DE MODELOS
# =============================
X = df_group.drop(columns=["cliente_id", "riesgo"])
y = df_group["riesgo"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelos_base = [
    ("rf", RandomForestClassifier(random_state=42)),
    ("etr", ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ("gb", GradientBoostingClassifier(random_state=42))
]

metricas_base = {}
print("\n📊 MÉTRICAS DE MODELOS BASE:")
for nombre, modelo in modelos_base:
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print(f"\n🔹 {nombre.upper()}")
    print(classification_report(y_test, y_pred, zero_division=0))
    metricas_base[nombre.upper()] = {
        "accuracy": reporte["accuracy"],
        "f1-score": reporte["weighted avg"]["f1-score"],
        "precision": reporte["weighted avg"]["precision"],
        "recall": reporte["weighted avg"]["recall"]
    }

# =============================
# 🔗 METAMODELO STACKING
# =============================
print("\n🔗 Entrenando metamodelo (RidgeClassifier)...")
stacking = StackingClassifier(
    estimators=modelos_base,
    final_estimator=RidgeClassifier(),
    n_jobs=-1
)
stacking.fit(X_train, y_train)

y_pred_meta = stacking.predict(X_test)
reporte_meta = classification_report(y_test, y_pred_meta, output_dict=True, zero_division=0)
print("\n✅ MÉTRICAS DEL METAMODELO:")
print(classification_report(y_test, y_pred_meta, zero_division=0))

metricas_meta = {
    "accuracy": reporte_meta["accuracy"],
    "f1-score": reporte_meta["weighted avg"]["f1-score"],
    "precision": reporte_meta["weighted avg"]["precision"],
    "recall": reporte_meta["weighted avg"]["recall"]
}

# =============================
# 💾 GUARDAR RESULTADOS
# =============================
df_group.to_excel("riesgo_clientes.xlsx", index=False)
joblib.dump({
    "modelo": stacking,
    "scaler": scaler,
    "metricas_base": metricas_base,
    "metricas_meta": metricas_meta
}, "modelo_riesgo_clientes.pkl")

print("\n✅ Modelo y archivo de riesgo por cliente generados correctamente.")
