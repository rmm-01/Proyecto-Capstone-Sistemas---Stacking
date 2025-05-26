import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# 1. Cargar archivos
df_stock = pd.read_excel("stock_limpio.xlsx")
df_demanda = pd.read_excel("demanda_limpia.xlsx")

# 2. Normalizar descripciones
df_stock["descripcion_norm"] = df_stock["descripcion"].astype(str).str.upper().str.strip()
df_demanda["descripcion_norm"] = df_demanda["oportunidad_/_descripcion"].astype(str).str.upper().str.strip()

# 3. Agregar métricas de demanda
agrupado = df_demanda.groupby("descripcion_norm").agg({
    "cantidad": "sum",
    "proyeccion_s/.": "mean"
}).rename(columns={"cantidad": "cantidad_total", "proyeccion_s/.": "proyeccion_prom"}).reset_index()

# 4. Fusionar con stock
df = df_stock.merge(agrupado, on="descripcion_norm", how="left")
df["cantidad_total"] = df["cantidad_total"].fillna(0)
df["proyeccion_prom"] = df["proyeccion_prom"].fillna(0)

# 5. Etiqueta objetivo: exceso si stock > percentil 85
umbral = df["stock"].quantile(0.85)
df["exceso_stock"] = (df["stock"] > umbral).astype(int)

# 6. Features y target (sin ratio)
X = df[["cantidad_total", "proyeccion_prom"]]
y = df["exceso_stock"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 7. Calcular pesos balanceados
pesos = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
clase_weights_dict = {int(i): peso for i, peso in enumerate(pesos)}

# 8. Modelos base
base_models = [
    ("rf", RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
    ("catboost", CatBoostClassifier(verbose=0, random_state=42, class_weights=clase_weights_dict)),
    ("extratrees", ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42))
]
meta_model = LogisticRegression(class_weight='balanced', max_iter=500)

# 9. Métricas modelos base
metricas_base = {}
for nombre, modelo in base_models:
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    metricas_base[nombre.upper()] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4)
    }

# 10. Metamodelo
stacking = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
stacking.fit(X_train, y_train)
y_pred_meta = stacking.predict(X_test)
metricas_meta = {
    "accuracy": round(accuracy_score(y_test, y_pred_meta), 4),
    "f1_score": round(f1_score(y_test, y_pred_meta), 4),
    "recall": round(recall_score(y_test, y_pred_meta), 4)
}

# 11. Guardar modelo y métricas
joblib.dump((stacking, {"base": metricas_base, "metamodelo": metricas_meta}), "modelo_exceso.pkl")

# 12. Mostrar métricas
print("📊 MÉTRICAS BASE:")
for k, v in metricas_base.items():
    print(f"{k}: {v}")
print("\n📊 METAMODELO:")
for k, v in metricas_meta.items():
    print(f"{k}: {v}")
