from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import numpy as np
import joblib
from limpieza_logistica import limpiar_data_demanda, limpiar_data_stock

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/outbound")
def outbound():
    return render_template("outbound.html")

@app.route("/limpieza")
def limpieza():
    return render_template("limpieza.html")

@app.route("/inventory")
def inventory():
    return render_template("inventory.html")

@app.route("/inbound")
def inbound():
    return render_template("inbound.html")


@app.route("/limpiar_datos", methods=["POST"])
def limpiar_datos():
    try:
        archivo_demanda = request.files.get("excel-file")
        archivo_stock = request.files.get("stock-file")

        anio_stock = request.form.get("anio_stock", type=int)
        mes_stock = request.form.get("mes_stock", type=int)
        anio_demanda = request.form.get("anio_demanda", type=int)

        if not archivo_demanda:
            return jsonify({"error": "El archivo de demanda es obligatorio"}), 400

        path_demanda = os.path.join(app.config['UPLOAD_FOLDER'], archivo_demanda.filename)
        archivo_demanda.save(path_demanda)

        if archivo_stock:
            path_stock = os.path.join(app.config['UPLOAD_FOLDER'], archivo_stock.filename)
            archivo_stock.save(path_stock)
        else:
            path_stock = None

        # Procesar demanda
        df_d = pd.read_excel(path_demanda, sheet_name="Data", dtype=str)
        limpio_d = limpiar_data_demanda(df_d, anio_manual=anio_demanda)
        limpio_d.to_excel("demanda_limpia.xlsx", index=False)

        vista_previa = limpio_d.head(10)
        vista_json = {
            "columns": list(vista_previa.columns),
            "rows": vista_previa.astype(str).values.tolist()
        }

        # Procesar stock
        vista_stock_json = None
        if path_stock:
            try:
                df_s = pd.read_excel(path_stock, sheet_name="General", dtype=str)
                limpio_s = limpiar_data_stock(df_s, anio_manual=anio_stock, mes_manual=mes_stock)
                limpio_s.to_excel("stock_limpio.xlsx", index=False)

                vista_stock = limpio_s.head(10)
                vista_stock_json = {
                    "columns": list(vista_stock.columns),
                    "rows": vista_stock.astype(str).values.tolist()
                }
            except Exception as e:
                return jsonify({"error": f"Error al limpiar stock: {str(e)}"}), 400

        return jsonify({
            "mensaje": "✔ Archivos limpiados con éxito.",
            "vista_previa": vista_json,
            "vista_stock": vista_stock_json
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/outbound/pronostico_clientes")
def mostrar_formulario_prediccion_clientes():
    return render_template("outbound_pronostico_clientes.html")

@app.route("/prediccion_clientes", methods=["POST"])
def prediccion_clientes():
    archivo = request.files.get("archivo")
    if not archivo:
        return "Debes subir un archivo .xlsx", 400

    df = pd.read_excel(archivo)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    for col in ["cliente_id", "monto", "mes", "anio", "cantidad"]:
        if col not in df.columns:
            return f"Falta la columna requerida: {col}", 400

    from sklearn.preprocessing import LabelEncoder

    df = df.dropna(subset=["cliente_id", "monto", "mes", "anio", "cantidad"])
    df = df[(df["cantidad"] >= 0) & (df["monto"] >= 0)]
    q99 = df["cantidad"].quantile(0.99)
    df = df[df["cantidad"] <= q99]

    # Codificar cliente_id con manejo de desconocidos
    le = joblib.load("encoder_cliente.pkl")
    clientes_vistos = set(le.classes_)
    df["cliente_id"] = df["cliente_id"].astype(str).apply(lambda x: x if x in clientes_vistos else "desconocido")

    if "desconocido" not in le.classes_:
        le.classes_ = np.append(le.classes_, "desconocido")

    df["cliente_id"] = le.transform(df["cliente_id"])

    # Escalamiento
    X_raw = df[["cliente_id", "monto", "mes", "anio"]]
    scaler = joblib.load("scaler_clientes.pkl")
    X = scaler.transform(X_raw)
    y = df["cantidad"]

    # Predicción
    stacking_model = joblib.load("modelo_clientes.pkl")
    y_pred_log = stacking_model.predict(X)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mape = None if (y == 0).any() else np.mean(np.abs((y - y_pred) / y)) * 100

    df_resultado = df[["cliente_id", "monto", "mes", "anio"]].copy()
    df_resultado["real"] = y.values
    df_resultado["prediccion"] = y_pred

    nombre_excel = "static/predicciones_clientes.xlsx"
    os.makedirs("static", exist_ok=True)
    df_resultado.to_excel(nombre_excel, index=False)

    return render_template(
        "outbound_pronostico_clientes.html",
        tabla=df_resultado.head(10).to_dict(orient="records"),
        metricas_base=joblib.load("metricas_clientes.pkl")["base"],
        metricas_meta=joblib.load("metricas_clientes.pkl")["metamodelo"],
        enlace="/" + nombre_excel
    )

@app.route("/outbound/segmentacion_clientes")
def vista_segmentacion_clientes():
    return render_template("outbound_segmentacion_clientes.html")

@app.route("/segmentacion_clientes", methods=["POST"])
def segmentacion_clientes():
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.preprocessing import StandardScaler

    archivo = request.files["archivo"]
    df = pd.read_excel(archivo)
    df.columns = df.columns.str.strip().str.lower()

    # Preprocesamiento
    df["cliente_id"] = df["cliente_id"].astype(str)
    df_group = df.groupby("cliente_id").agg({
        "monto": ["sum", "mean"],
        "cantidad": ["sum", "mean"],
        "anio": "nunique",
        "mes": "nunique"
    })
    df_group.columns = ["monto_total", "monto_promedio", "cantidad_total", "cantidad_promedio", "anios_activo", "meses_activo"]
    df_group.reset_index(inplace=True)

    # Filtrar clientes pasivos
    df_group = df_group[df_group["cantidad_total"] > 10]

    # Escalar
    X = df_group.drop(columns=["cliente_id"])
    scaler = joblib.load("scaler_segmentacion.pkl")
    X_scaled = scaler.transform(X)

    # Cargar modelo
    modelo = joblib.load("modelo_segmentacion.pkl")
    segmentos = modelo.predict(X_scaled)

    df_group["segmento"] = segmentos
    df_group.to_excel("segmentacion_clientes.xlsx", index=False)

    # Cargar métricas
    metricas = joblib.load("metricas_segmentacion.pkl")
    metricas_base = metricas["base"]
    metricas_meta = metricas["metamodelo"]
    silhouette = metricas["silhouette"]

    tabla = df_group.head(10).to_dict(orient="records")

    return render_template(
        "outbound_segmentacion_clientes.html",
        tabla=tabla,
        metricas_base=metricas_base,
        metricas_meta=metricas_meta,
        silhouette=silhouette,
        enlace="/descargar/segmentacion_clientes.xlsx"
    )

@app.route("/outbound/pronostico_materiales")
def mostrar_formulario_prediccion_materiales():
    return render_template("outbound_pronostico_materiales.html")

@app.route("/prediccion_materiales", methods=["POST"])
def prediccion_materiales():
    import pandas as pd
    import numpy as np
    import joblib
    from datetime import datetime

    archivo_demanda = request.files["archivo_demanda"]
    archivo_stock = request.files["archivo_stock"]
    df_demanda = pd.read_excel(archivo_demanda)
    df_stock = pd.read_excel(archivo_stock)

    # Preprocesamiento
    df_demanda.columns = df_demanda.columns.str.strip().str.lower()
    df_stock.columns = df_stock.columns.str.strip().str.lower()

    df_demanda["clave"] = df_demanda["dim1"].astype(str) + "_" + df_demanda["unidad"].astype(str)
    df_stock["clave"] = df_stock["dim_1"].astype(str) + "_" + df_stock["unidad"].astype(str)

    df = pd.merge(df_demanda, df_stock[["clave", "stock"]].drop_duplicates(), on="clave", how="left")
    df.rename(columns={"stock": "stock_relacionado"}, inplace=True)

    df = df.dropna(subset=["cantidad", "stock_relacionado", "monto", "fecha"])
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df[(df["cantidad"] >= 0) & (df["monto"] >= 0)]
    df = df[df["cantidad"] <= df["cantidad"].quantile(0.99)]

    # Features
    X = df[["monto", "stock_relacionado"]].copy()
    X["mes"] = df["fecha"].dt.month
    X["anio"] = df["fecha"].dt.year
    y_real = df["cantidad"]

    # Escalamiento
    scaler = joblib.load("scaler_materiales.pkl")
    X_scaled = scaler.transform(X)

    # Modelo y predicción
    modelo = joblib.load("modelo_materiales.pkl")
    y_pred = np.expm1(modelo.predict(X_scaled))

    # Crear tabla
    df_resultado = X.copy()
    df_resultado["real"] = y_real.values
    df_resultado["prediccion"] = y_pred

    # Guardar Excel
    nombre_excel = "predicciones_materiales.xlsx"
    df_resultado.to_excel(nombre_excel, index=False)

    # Cargar métricas guardadas
    metricas = joblib.load("metricas_materiales.pkl")
    metricas_base = metricas["base"]
    metricas_meta = metricas["metamodelo"]

    # Tabla para HTML (primeras 10 filas)
    tabla = df_resultado.head(10).to_dict(orient="records")

    return render_template(
        "outbound_pronostico_materiales.html",
        tabla=tabla,
        enlace=f"/descargar/{nombre_excel}",
        metricas_base=metricas_base,
        metricas_meta=metricas_meta
    )

@app.route("/inbound/riesgo_clientes", methods=["GET"])
def vista_riesgo_clientes():
    return render_template("inbound_riesgo_clientes.html")

@app.route("/riesgo_clientes", methods=["POST"])
def riesgo_clientes():
    import pandas as pd
    import joblib

    archivo = request.files["archivo"]
    df = pd.read_excel(archivo)
    df.columns = df.columns.str.strip().str.lower()
    df = df[df["cantidad"] > 0]
    df["cliente_id"] = df["cliente_id"].astype(str)

    # Agregación por cliente
    df_group = df.groupby("cliente_id").agg({
        "monto": ["sum", "mean", "std"],
        "cantidad": ["sum", "mean", "std"]
    })
    df_group.columns = ["monto_total", "monto_promedio", "monto_std",
                        "cantidad_total", "cantidad_promedio", "cantidad_std"]
    df_group.reset_index(inplace=True)
    df_group.fillna(0, inplace=True)
    df_group["monto_unitario_prom"] = df_group["monto_total"] / df_group["cantidad_total"]

    X = df_group.drop(columns=["cliente_id"])

    # Cargar modelo
    modelo_riesgo = joblib.load("modelo_riesgo_clientes.pkl")
    scaler = modelo_riesgo["scaler"]
    modelo = modelo_riesgo["modelo"]
    metricas_base = modelo_riesgo["metricas_base"]
    metricas_meta = modelo_riesgo["metricas_meta"]

    X_scaled = scaler.transform(X)
    predicciones = modelo.predict(X_scaled)
    df_group["riesgo"] = predicciones
    df_group.to_excel("riesgo_clientes.xlsx", index=False)

    tabla = df_group.head(10).to_dict(orient="records")

    # 🔧 Reformatear métricas base
    metricas_base_format = {}
    for modelo, valores in metricas_base.items():
        metricas_base_format[modelo] = {
            "Accuracy": round(valores.get("accuracy", 0), 4),
            "F1 Score": round(valores.get("f1-score", 0), 4),
            "Precision": round(valores.get("precision", 0), 4),
            "Recall": round(valores.get("recall", 0), 4)
        }

    # 🔧 Reformatear métricas del metamodelo
    metricas_meta_format = {
        "Accuracy": round(metricas_meta.get("accuracy", 0), 4),
        "F1 Score": round(metricas_meta.get("f1-score", 0), 4),
        "Precision": round(metricas_meta.get("precision", 0), 4),
        "Recall": round(metricas_meta.get("recall", 0), 4)
    }

    return render_template("inbound_riesgo_clientes.html",
                           tabla=tabla,
                           enlace="/descargar/riesgo_clientes.xlsx",
                           metricas_base=metricas_base_format,
                           metricas_meta=metricas_meta_format)


@app.route("/inventory/pronostico_exceso_stock")
def vista_pronostico_exceso_stock():
    return render_template("inventory_exceso_stock.html")

@app.route("/prediccion_exceso_stock", methods=["POST"])
def prediccion_exceso_stock():
    from difflib import get_close_matches

    archivo_demanda = request.files.get("archivo_demanda")
    archivo_stock = request.files.get("archivo_stock")

    if not archivo_demanda or not archivo_stock:
        return "❌ Debes subir ambos archivos: demanda y stock (.xlsx).", 400

    df_demanda = pd.read_excel(archivo_demanda)
    df_stock = pd.read_excel(archivo_stock)

    # Normalización de columnas
    df_demanda.columns = df_demanda.columns.str.lower().str.strip()
    df_stock.columns = df_stock.columns.str.lower().str.strip()
    df_demanda["oportunidad_/_descripcion"] = df_demanda["oportunidad_/_descripcion"].astype(str).str.upper().str.strip()
    df_stock["descripcion"] = df_stock["descripcion"].astype(str).str.upper().str.strip()

    # Mapeo de descripciones usando similitud textual
    map_relacion = {}
    for desc in df_demanda["oportunidad_/_descripcion"].unique():
        match = get_close_matches(desc, df_stock["descripcion"].unique(), n=1, cutoff=0.4)
        if match:
            map_relacion[desc] = match[0]

    df_demanda["descripcion_relacionada"] = df_demanda["oportunidad_/_descripcion"].map(map_relacion)

    # Extraer variables desde demanda
    agrupado = df_demanda.groupby("descripcion_relacionada").agg({
        "cantidad": "sum",
        "proyeccion_s/.": "mean"
    }).rename(columns={"cantidad": "cantidad_total", "proyeccion_s/.": "proyeccion_prom"}).reset_index()

    # Unir con el stock original
    df_stock["descripcion_relacionada"] = df_stock["descripcion"]
    df = pd.merge(df_stock, agrupado, on="descripcion_relacionada", how="left")
    df["cantidad_total"] = df["cantidad_total"].fillna(0)
    df["proyeccion_prom"] = df["proyeccion_prom"].fillna(0)

    if df.empty:
        return "❌ No se pudieron cruzar los datos entre demanda y stock. Verifica las descripciones.", 400

    # Cargar modelo entrenado
    modelo, metricas = joblib.load("modelo_exceso.pkl")

    # Predicción
    X = df[["cantidad_total", "proyeccion_prom"]]
    df["exceso_stock_predicho"] = modelo.predict(X)

    # Guardar resultados
    ruta_resultado = "static/prediccion_exceso_stock.xlsx"
    df.to_excel(ruta_resultado, index=False)

    return render_template("inventory_exceso_stock.html",
        tabla=df.head(10).to_dict(orient="records"),
        metricas_base=metricas["base"],
        metricas_meta=metricas["metamodelo"],
        enlace="/" + ruta_resultado
    )

@app.route('/inventory/quiebre_stock')
def quiebre_stock():
    try:
        # Cargar datos limpios
        stock = pd.read_excel("stock_limpio.xlsx")
        demanda = pd.read_excel("demanda_limpia.xlsx")

        stock.columns = stock.columns.str.strip().str.lower().str.replace(" ", "_")
        demanda.columns = demanda.columns.str.strip().str.lower().str.replace(" ", "_")

        # Crear claves simplificadas
        stock["clave"] = stock["dim_1"].astype(str) + "_" + stock["unidad"].astype(str)
        demanda["clave"] = demanda["dim1"].astype(str) + "_" + demanda["unidad"].astype(str)

        if "anio" not in demanda.columns:
            demanda["anio"] = pd.to_datetime(demanda["fecha"]).dt.year
        if "anio" not in stock.columns:
            raise ValueError("El archivo de stock debe contener la columna 'anio'.")

        stock["clave_tiempo"] = stock["clave"] + "_" + stock["anio"].astype(str)
        demanda["clave_tiempo"] = demanda["clave"] + "_" + demanda["anio"].astype(str)

        # Agregaciones
        demanda_agg = demanda.groupby("clave_tiempo")["cantidad"].sum().reset_index()
        stock_agg = stock.groupby("clave_tiempo").mean(numeric_only=True).reset_index()

        df = pd.merge(stock_agg, demanda_agg, on="clave_tiempo", how="inner")
        df["quiebre_stock"] = (df["cantidad"] > df["stock"]).astype(int)

        columnas = ["stock", "por_llegar", "ov_1", "ov_2", "sitio"]
        for col in columnas:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)

        df_model = df[columnas + ["quiebre_stock"]].dropna()

        # Cargar modelo
        modelo = joblib.load("modelo_quiebre.pkl")
        X = df_model[columnas]
        df_model["prediccion"] = modelo.predict(X)

        # Evaluar (con los datos usados en predicción porque no hay train/test aquí)
        y_true = df_model["quiebre_stock"]
        y_pred = df_model["prediccion"]

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0)
        }

        return render_template("quiebre_stock.html",
                               tabla=df_model.head(10).to_html(classes="table table-bordered", index=False),
                               metrics=metrics)
    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
