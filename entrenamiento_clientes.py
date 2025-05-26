import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ============================
# 📥 CARGA Y PREPROCESAMIENTO
# ============================
print("📥 Cargando datos...")
df = pd.read_excel("demanda_limpia.xlsx")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Filtrar outliers y valores erróneos
df = df[df["cantidad"] >= 0]
df = df[df["cantidad"] <= df["cantidad"].quantile(0.99)]
df = df.dropna(subset=["cliente_id", "monto", "mes", "anio", "cantidad"])

# Codificación
le = LabelEncoder()
df["cliente_id"] = le.fit_transform(df["cliente_id"].astype(str))
joblib.dump(le, "encoder_cliente.pkl")

# ============================
# 🔄 FEATURES Y ESCALAMIENTO
# ============================
X = df[["cliente_id", "monto", "mes", "anio"]]
y = df["cantidad"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_clientes.pkl")

# ============================
# ✂ DIVISIÓN DE DATOS
# ============================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ============================
# 🧮 TRANSFORMACIÓN TARGET
# ============================
y_train_log = np.log1p(y_train)

# ============================
# 🤖 MODELOS BASE
# ============================
base_models = [
    ("rf", RandomForestRegressor(n_estimators=100, random_state=0)),
    ("xgb", XGBRegressor(n_estimators=100, verbosity=0, random_state=0)),
    ("knn", KNeighborsRegressor())
]

print("\n📊 MÉTRICAS DE MODELOS BASE:")
for name, model in base_models:
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    mape = None if (y_test == 0).any() else np.mean(np.abs((y_test - pred) / y_test)) * 100

    print(f"{name.upper():<5} | MAE: {mae:.2f} | R²: {r2:.4f} | MAPE: {mape:.2f}%" if mape else f"{name.upper():<5} | MAE: {mae:.2f} | R²: {r2:.4f} | MAPE: ⚠")

# ============================
# 🧠 METAMODELO STACKING
# ============================
print("\n🔗 ENTRENANDO METAMODELO (STACKING)...")
meta_model = Ridge()
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    n_jobs=-1  # ✅ Uso de todos los núcleos disponibles
)
stacking_model.fit(X_train, y_train_log)
joblib.dump(stacking_model, "modelo_clientes.pkl")

# ============================
# 📈 EVALUACIÓN FINAL
# ============================
y_pred_log = stacking_model.predict(X_test)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = None if (y_test == 0).any() else np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n📌 METAMODELO FINAL:")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%" if mape is not None else "MAPE: ⚠")
print(f"R²: {r2:.4f}")

# ============================
# 💾 GUARDAR RESULTADOS
# ============================
df_resultado = pd.DataFrame(X_test, columns=["cliente_id", "monto", "mes", "anio"])
df_resultado["real"] = y_test.values
df_resultado["prediccion"] = y_pred
df_resultado.to_excel("predicciones_clientes.xlsx", index=False)
print("\n✅ Script completado y resultados guardados.")

# ============================
# 💾 GUARDAR MÉTRICAS PARA FLASK
# ============================
metricas_base = {}
for name, model in base_models:
    model.fit(X_train, y_train_log)
    pred_log = model.predict(X_test)
    pred = np.expm1(pred_log)

    mae_b = mean_absolute_error(y_test, pred)
    r2_b = r2_score(y_test, pred)
    mape_b = None if (y_test == 0).any() else np.mean(np.abs((y_test - pred) / y_test)) * 100

    metricas_base[name.upper()] = {
        "MAE": round(mae_b, 2),
        "R2": round(r2_b, 4),
        "MAPE": round(mape_b, 2) if mape_b is not None else "⚠"
    }

metricas = {
    "base": metricas_base,
    "metamodelo": {
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2) if mape is not None else "⚠",
        "R2": round(r2, 4)
    }
}
joblib.dump(metricas, "metricas_clientes.pkl")
print("📁 Métricas guardadas en 'metricas_clientes.pkl'")
