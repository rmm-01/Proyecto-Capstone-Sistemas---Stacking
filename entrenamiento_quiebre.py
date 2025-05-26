import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def entrenamiento_quiebre(stock_file: str, demanda_file: str, output_model: str = "modelo_quiebre.pkl"):
    stock = pd.read_excel(stock_file)
    demanda = pd.read_excel(demanda_file)

    stock.columns = stock.columns.str.strip().str.lower().str.replace(" ", "_")
    demanda.columns = demanda.columns.str.strip().str.lower().str.replace(" ", "_")

    stock["clave"] = stock["dim_1"].astype(str) + "_" + stock["unidad"].astype(str)
    demanda["clave"] = demanda["dim1"].astype(str) + "_" + demanda["unidad"].astype(str)

    if "anio" not in demanda.columns:
        demanda["anio"] = pd.to_datetime(demanda["fecha"]).dt.year
    if "anio" not in stock.columns:
        raise ValueError("El archivo de stock debe contener la columna 'anio'.")

    stock["clave_tiempo"] = stock["clave"] + "_" + stock["anio"].astype(str)
    demanda["clave_tiempo"] = demanda["clave"] + "_" + demanda["anio"].astype(str)

    demanda_agg = demanda.groupby("clave_tiempo")["cantidad"].sum().reset_index()
    stock_agg = stock.groupby("clave_tiempo").mean(numeric_only=True).reset_index()

    df = pd.merge(stock_agg, demanda_agg, on="clave_tiempo", how="inner")
    df["quiebre_stock"] = np.where(df["cantidad"] > df["stock"], 1, 0)

    print("\n🔍 Diagnóstico del DataFrame resultante:")
    print("Registros después del merge:", len(df))
    print(df[["clave_tiempo", "stock", "cantidad"]].head(10))

    columnas_base = ["stock"]
    columnas_opcionales = ["por_entregar", "por_llegar", "ov_1", "ov_2", "sitio"]
    columnas_presentes = columnas_base + [col for col in columnas_opcionales if col in df.columns]

    df_model = df[columnas_presentes + ["quiebre_stock"]].copy()
    for col in columnas_opcionales:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(0)
    df_model = df_model.dropna(subset=columnas_base)

    print("\n✅ Columnas utilizadas:", columnas_presentes)
    print("✅ Registros completos disponibles:", len(df_model))
    print(df_model.head())

    if df_model.shape[0] < 10:
        print(f"❌ No hay suficientes registros completos para entrenar el modelo. Solo hay {df_model.shape[0]}")
        return

    X = df_model[columnas_presentes]
    y = df_model["quiebre_stock"]

    print("\n📊 Distribución de clases (quiebre_stock):")
    print(y.value_counts())
    if y.value_counts().min() < 5:
        print("⚠️ ADVERTENCIA: Una de las clases tiene menos de 5 ejemplos. Las métricas pueden no ser confiables.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # MODELOS BASE
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    lgb = LGBMClassifier(random_state=42)
    dt = DecisionTreeClassifier(random_state=42)

    # ENTRENAR Y EVALUAR MODELOS BASE
    def evaluar(modelo, X, y, nombre):
        pred = modelo.predict(X)
        print(f"\n🔹 MÉTRICAS PARA {nombre}")
        print(f"Accuracy:  {accuracy_score(y, pred):.4f}")
        print(f"Precision: {precision_score(y, pred, zero_division=0):.4f}")
        print(f"Recall:    {recall_score(y, pred, zero_division=0):.4f}")
        print(f"F1 Score:  {f1_score(y, pred, zero_division=0):.4f}")
        print("-" * 30)

    xgb.fit(X_train, y_train)
    evaluar(xgb, X_test, y_test, "XGBoost")

    lgb.fit(X_train, y_train)
    evaluar(lgb, X_test, y_test, "LightGBM")

    dt.fit(X_train, y_train)
    evaluar(dt, X_test, y_test, "Decision Tree")

    # METAMODELO (STACKING)
    base_estimators = [("xgb", xgb), ("lgb", lgb), ("dt", dt)]
    meta = LogisticRegression(max_iter=1000)

    stack = StackingClassifier(estimators=base_estimators, final_estimator=meta, passthrough=True)
    stack.fit(X_train, y_train)

    evaluar(stack, X_test, y_test, "Stacking (XGBoost + LightGBM + DT)")

    joblib.dump(stack, output_model)
    print(f"\n✅ Modelo guardado en: {output_model}")

if __name__ == "__main__":
    entrenamiento_quiebre(
        stock_file="stock_limpio.xlsx",
        demanda_file="demanda_limpia.xlsx",
        output_model="modelo_quiebre.pkl"
    )
