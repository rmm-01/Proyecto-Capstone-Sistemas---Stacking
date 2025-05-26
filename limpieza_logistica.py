import pandas as pd
import unicodedata
from datetime import datetime

# --------------------------------------
# FUNCIONES DE UTILIDAD
# --------------------------------------

def estandarizar_columnas(columnas):
    return [
        unicodedata.normalize('NFKD', c).encode('ascii', 'ignore').decode('utf-8').lower().strip().replace(" ", "_")
        for c in columnas
    ]

def normalizar_texto(texto):
    if isinstance(texto, str):
        return unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8').lower().strip()
    return texto

# --------------------------------------
# LIMPIEZA DEMANDA
# --------------------------------------

def limpiar_data_demanda(df, anio_manual=None, mes_manual=None):
    try:
        df = df.copy()
        df.columns = estandarizar_columnas(df.columns)

        columnas_renombrar = {
            'cliente': 'cliente_id',
            'monto_s/.': 'monto',
            'cantidad': 'cantidad',
            'fecha_de_registro': 'fecha'
        }
        df = df.rename(columns={col: columnas_renombrar[col] for col in columnas_renombrar if col in df.columns})

        df['monto'] = pd.to_numeric(df.get('monto'), errors='coerce')
        df['cantidad'] = pd.to_numeric(df.get('cantidad'), errors='coerce')

        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            df['mes'] = df['fecha'].dt.month
            df['anio'] = df['fecha'].dt.year

            if anio_manual:
                df.loc[df['anio'].isna(), 'anio'] = anio_manual
                df.loc[df['mes'].isna(), 'mes'] = mes_manual if mes_manual else 1
        else:
            df['anio'] = anio_manual if anio_manual else None
            df['mes'] = mes_manual if mes_manual else None

        # 🔧 Normalizar campos
        df['cliente_id'] = df['cliente_id'].apply(normalizar_texto)

        # ❌ Eliminar columnas irrelevantes
        columnas_a_excluir = ['telefono', 'observaciones', 'facturacion', 'ejecutivo_comercial']
        df = df.drop(columns=[col for col in columnas_a_excluir if col in df.columns], errors='ignore')

        # ❗ Eliminar duplicados exactos
        df = df.drop_duplicates(subset=['cliente_id', 'monto', 'cantidad', 'anio', 'mes'])

        print("🔍 Diagnóstico demanda antes del filtrado:")
        print(df[['cliente_id', 'monto', 'cantidad', 'anio']].isnull().sum())
        print(f"Total filas antes: {len(df)}")

        columnas_obligatorias = ['cliente_id', 'monto', 'cantidad', 'anio']
        df_limpio = df.dropna(subset=columnas_obligatorias)

        print(f"✅ Total filas después: {len(df_limpio)}")
        print(f"❌ Filas eliminadas: {len(df) - len(df_limpio)}")

        return df_limpio

    except Exception as e:
        raise RuntimeError(f"❌ Error en limpieza de demanda: {str(e)}")

# --------------------------------------
# LIMPIEZA STOCK
# --------------------------------------

def limpiar_data_stock(df, anio_manual=None, mes_manual=None):
    try:
        df = df.copy()
        df.columns = estandarizar_columnas(df.columns)

        columnas_renombrar = {
            'codigo_de_articulo': 'material',
            'disponible': 'stock'
        }
        df = df.rename(columns={col: columnas_renombrar[col] for col in columnas_renombrar if col in df.columns})

        df['stock'] = pd.to_numeric(df.get('stock'), errors='coerce')

        if 'fecha' in df.columns and df['fecha'].notnull().sum() > 0:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            df['mes'] = df['fecha'].dt.month
            df['anio'] = df['fecha'].dt.year
        elif anio_manual:
            df['anio'] = anio_manual
            df['mes'] = mes_manual if mes_manual else 1
        else:
            df['anio'] = None
            df['mes'] = None

        # 🔧 Normalizar materiales
        df['material'] = df['material'].apply(normalizar_texto)

        # ❗ Eliminar duplicados exactos
        df = df.drop_duplicates(subset=['material', 'anio', 'stock'])

        # ❌ Filtrar registros sin stock
        df = df[df['stock'] > 0]

        print("🔍 Diagnóstico stock antes del filtrado:")
        print(df[['material', 'stock', 'anio']].isnull().sum())
        print(f"Total filas antes: {len(df)}")

        columnas_obligatorias = ['material', 'stock', 'anio']
        df_limpio = df.dropna(subset=columnas_obligatorias)

        print(f"✅ Total filas después: {len(df_limpio)}")
        print(f"❌ Filas eliminadas: {len(df) - len(df_limpio)}")

        return df_limpio

    except Exception as e:
        raise RuntimeError(f"❌ Error en limpieza de stock: {str(e)}")

# --------------------------------------
# BLOQUE DE PRUEBA DIRECTA
# --------------------------------------

if __name__ == "__main__":
    try:
        print("\n🧪 Ejecutando limpieza de DEMANDA...")
        archivo_demanda = "Copia de BASES DE DATOS GENERAL 2024 (005).xlsx"
        hoja = "Data"
        df_demanda = pd.read_excel(archivo_demanda, sheet_name=hoja)
        df_demanda_limpio = limpiar_data_demanda(df_demanda, anio_manual=2024)
        df_demanda_limpio.to_excel("demanda_limpia.xlsx", index=False)
        print("✅ Archivo 'demanda_limpia.xlsx' guardado.\n")

        print("\n🧪 Ejecutando limpieza de STOCK...")
        archivo_stock = "Copia de Stock hasta 02.04.xlsx"
        hoja_stock = "General"
        df_stock = pd.read_excel(archivo_stock, sheet_name=hoja_stock)
        df_stock_limpio = limpiar_data_stock(df_stock, anio_manual=2024)
        df_stock_limpio.to_excel("stock_limpio.xlsx", index=False)
        print("✅ Archivo 'stock_limpio.xlsx' guardado.")

    except Exception as e:
        print(f"❌ Error en ejecución principal: {str(e)}")
