import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# ============================
# 📥 CARGA Y PREPROCESAMIENTO
# ============================
print("📥 Cargando data limpia...")
df_demanda = pd.read_excel("demanda_limpia.xlsx")
df_stock = pd.read_excel("stock_limpio.xlsx")

df_demanda.columns = df_demanda.columns.str.strip().str.lower()
df_stock.columns = df_stock.columns.str.strip().str.lower()

df_demanda["clave"] = df_demanda["dim1"].astype(str) + "_" + df_demanda["unidad"].astype(str)
df_stock["clave"] = df_stock["dim_1"].astype(str) + "_" + df_stock["unidad"].astype(str)

df = pd.merge(df_demanda, df_stock[["clave", "stock"]].drop_duplicates(), on="clave", how="left")
df.rename(columns={"stock": "stock_relacionado"}, inplace=True)

# ============================
# 🧹 LIMPIEZA
# ============================
print("🧹 Limpiando datos...")
df = df.dropna(subset=["cantidad", "stock_relacionado", "monto", "fecha"])
df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
df["monto"] = pd.to_numeric(df["monto"], errors="coerce")
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df = df[(df["cantidad"] >= 0) & (df["monto"] >= 0)]
df = df[df["cantidad"] <= df["cantidad"].quantile(0.99)]

# ============================
# 🎯 FEATURES Y TARGET
# ============================
X = df[["monto", "stock_relacionado"]].copy()
X["mes"] = df["fecha"].dt.month
X["anio"] = df["fecha"].dt.year
y = df["cantidad"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler_materiales.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_train_log = np.log1p(y_train)

# ============================
# 🤖 MODELOS BASE (con ExtraTrees)
# ============================
modelos_base = [
    ("lgbm", LGBMRegressor()),
    ("xgb", XGBRegressor(verbosity=0)),
    ("etr", ExtraTreesRegressor(n_estimators=100, random_state=42))
]

metricas_base = {}
print("\n📊 MÉTRICAS DE MODELOS BASE:")
for nombre, modelo in modelos_base:
    modelo.fit(X_train, y_train_log)
    pred_log = modelo.predict(X_test)
    pred = np.expm1(pred_log)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    metricas_base[nombre.upper()] = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }

    print(f"{nombre.upper():<6} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

# ============================
# 🧠 METAMODELO STACKING
# ============================
print("\n🔗 Entrenando metamodelo (Stacking)...")
meta_modelo = RandomForestRegressor(random_state=42)
stacking = StackingRegressor(estimators=modelos_base, final_estimator=meta_modelo, n_jobs=-1)
stacking.fit(X_train, y_train_log)
joblib.dump(stacking, "modelo_materiales.pkl")

# ============================
# 📈 EVALUACIÓN FINAL
# ============================
y_pred_log = stacking.predict(X_test)
y_pred = np.expm1(y_pred_log)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

metricas_meta = {
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "R2": round(r2, 4)
}

print("\n🤖 MÉTRICAS DEL METAMODELO (STACKING):")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

# ============================
# 💾 GUARDAR RESULTADOS
# ============================
df_resultado = pd.DataFrame(X_test, columns=["monto", "stock_relacionado", "mes", "anio"])
df_resultado["real"] = y_test.values
df_resultado["prediccion"] = y_pred
df_resultado.to_excel("predicciones_materiales.xlsx", index=False)

joblib.dump({
    "base": metricas_base,
    "metamodelo": metricas_meta
}, "metricas_materiales.pkl")

print("\n✅ Script finalizado. Modelo y métricas guardadas.")
