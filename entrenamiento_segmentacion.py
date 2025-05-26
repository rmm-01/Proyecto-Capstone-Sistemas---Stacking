import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import silhouette_score

# =============================
# 📥 CARGA Y AGRUPACIÓN
# =============================
print("📥 Cargando demanda limpia...")
df = pd.read_excel("demanda_limpia.xlsx")
df.columns = df.columns.str.strip().str.lower()
df = df[["cliente_id", "monto", "cantidad", "anio", "mes"]]
df["cliente_id"] = df["cliente_id"].astype(str)

df_group = df.groupby("cliente_id").agg({
    "monto": ["sum", "mean"],
    "cantidad": ["sum", "mean"],
    "anio": "nunique",
    "mes": "nunique"
})
df_group.columns = ["monto_total", "monto_promedio", "cantidad_total", "cantidad_promedio", "anios_activo", "meses_activo"]
df_group.reset_index(inplace=True)

# =============================
# ⚠️ FILTRAR CLIENTES PASIVOS
# =============================
df_group = df_group[df_group["cantidad_total"] > 10]
print(f"Clientes después del filtro: {len(df_group)}")

# =============================
# 🔢 ESCALAMIENTO Y CLUSTERING
# =============================
print("🔀 Segmentando con KMeans (n=2)...")
X_cluster = df_group.drop("cliente_id", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df_group["segmento"] = kmeans.fit_predict(X_scaled)

score_silhouette = silhouette_score(X_scaled, df_group["segmento"])
print(f"Silhouette Score: {score_silhouette:.3f}")
joblib.dump(scaler, "scaler_segmentacion.pkl")

# =============================
# 🤖 ENTRENAMIENTO SUPERVISADO
# =============================
X = df_group.drop(columns=["cliente_id", "segmento"])
y = df_group["segmento"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelos_base = [
    ("xgb", XGBClassifier(verbosity=0)),
    ("rf", RandomForestClassifier(random_state=42)),
    ("etr", ExtraTreesClassifier(n_estimators=100, random_state=42))
]

metricas_base = {}
print("\n📊 MÉTRICAS DE MODELOS BASE:")
for nombre, modelo in modelos_base:
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")

    metricas_base[nombre.upper()] = {
        "Accuracy": round(acc, 4),
        "F1 Score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4)
    }

    print(f"{nombre.upper()} | Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

# =============================
# 🔗 METAMODELO STACKING
# =============================
print("\n🔗 Entrenando metamodelo (Logistic Regression)...")

stacking = StackingClassifier(
    estimators=modelos_base,
    final_estimator=LogisticRegression(),
    n_jobs=-1
)
stacking.fit(X_train, y_train)
joblib.dump(stacking, "modelo_segmentacion.pkl")

# Evaluación metamodelo
y_pred = stacking.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
prec = precision_score(y_test, y_pred, average="macro")
rec = recall_score(y_test, y_pred, average="macro")

metricas_meta = {
    "Accuracy": round(acc, 4),
    "F1 Score": round(f1, 4),
    "Precision": round(prec, 4),
    "Recall": round(rec, 4)
}

print(f"\n✅ META Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

# =============================
# 💾 GUARDAR RESULTADOS
# =============================
df_group.to_excel("segmentacion_clientes.xlsx", index=False)
joblib.dump({
    "base": metricas_base,
    "metamodelo": metricas_meta,
    "silhouette": round(score_silhouette, 4)
}, "metricas_segmentacion.pkl")

print("\n✅ Entrenamiento completado y archivos guardados.")
