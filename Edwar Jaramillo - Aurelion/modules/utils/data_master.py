# modules/utils/data_master.py
import streamlit as st
import pandas as pd
import numpy as np

def construir_tabla_maestra(datasets, mostrar_mensajes=True, enriquecer=True):
    """
    Combina productos, clientes, ventas y detalle_ventas en una sola tabla analítica consolidada,
    con posibilidad de generar métricas derivadas y enriquecimiento automático.

    Args:
        datasets (dict): Diccionario con DataFrames cargados en memoria.
        mostrar_mensajes (bool): Si True, muestra mensajes en Streamlit.
        enriquecer (bool): Si True, agrega métricas y columnas derivadas para análisis avanzado.

    Returns:
        pd.DataFrame: DataFrame consolidado (tabla maestra) con métricas derivadas.
    """
    try:
        productos = datasets.get("productos", pd.DataFrame())
        clientes = datasets.get("clientes", pd.DataFrame())
        ventas = datasets.get("ventas", pd.DataFrame())
        detalle = datasets.get("detalle_ventas", pd.DataFrame())

        # === 1️⃣ Validación inicial ===
        if any(df.empty for df in [productos, clientes, ventas, detalle]):
            if mostrar_mensajes:
                st.warning("⚠️ No se puede crear la tabla maestra: falta al menos una tabla base.")
            return pd.DataFrame()

        # === 2️⃣ Selección explícita de columnas relevantes ===
        productos_cols = [c for c in productos.columns if c in ["id_producto", "nombre_producto", "categoria", "precio_unitario"]]
        clientes_cols = [c for c in clientes.columns if c in ["id_cliente", "nombre_cliente", "email", "ciudad", "fecha_alta"]]
        ventas_cols = [c for c in ventas.columns if c in ["id_venta", "id_cliente", "medio_pago", "fecha", "nombre_cliente"]]
        detalle_cols = [c for c in detalle.columns if c in ["id_venta", "id_producto", "cantidad", "precio_unitario", "importe"]]

        # Reducir las tablas a las columnas importantes
        productos = productos[productos_cols]
        clientes = clientes[clientes_cols]
        ventas = ventas[ventas_cols]
        detalle = detalle[detalle_cols]

        # === 3️⃣ Unión progresiva ===
        maestra = (
            detalle
            .merge(ventas, on="id_venta", how="left", suffixes=("", "_venta"))
            .merge(productos, on="id_producto", how="left", suffixes=("", "_producto"))
            .merge(clientes, on="id_cliente", how="left", suffixes=("", "_cliente"))
        )
        # === 4️⃣ Calcular o validar 'importe_total' ===
        if "cantidad" in maestra.columns and "precio_unitario" in maestra.columns:
            maestra["importe_total"] = maestra["cantidad"] * maestra["precio_unitario"]
        else:
            maestra["importe_total"] = np.nan
            if mostrar_mensajes:
                st.warning("⚠️ No se detectaron columnas válidas para calcular 'importe_total'.")

        # === 5️⃣ Conversión de fechas ===
        posibles_fechas = [c for c in maestra.columns if "fecha" in c.lower()]
        for col in posibles_fechas:
            maestra[col] = pd.to_datetime(maestra[col], errors="coerce")

        # === 6️⃣ Enriquecimiento opcional ===
        if enriquecer and "_enriquecer_tabla_maestra" in globals():
            maestra = _enriquecer_tabla_maestra(maestra, mostrar_mensajes)

        # === 7️⃣ Limpieza final ===
        maestra = maestra.drop_duplicates()

        # Validación del campo 'categoria'
        if "categoria" not in maestra.columns:
            maestra["categoria"] = "Sin categoría"
            if mostrar_mensajes:
                st.warning("⚠️ No se encontró columna 'categoria' en productos; se agregó por defecto.")

        if mostrar_mensajes:
            st.success(f"✅ Tabla maestra creada correctamente con {len(maestra)} registros y {len(maestra.columns)} columnas.")

        return maestra

    except Exception as e:
        if mostrar_mensajes:
            st.error(f"❌ Error al construir la tabla maestra: {e}")
        return pd.DataFrame()


# ============================================================
# 🔍 FUNCIÓN INTERNA: ENRIQUECIMIENTO AUTOMÁTICO
# ============================================================

def _enriquecer_tabla_maestra(df, mostrar_mensajes=True):
    """
    Enriquecer la tabla maestra con métricas analíticas derivadas:
    - Variables temporales (año, mes, trimestre)
    - Totales y participaciones por cliente, categoría y producto
    - Identificación de productos con baja rotación o ventas atípicas
    - Ticket promedio mensual y global
    """

    try:
        # === 🗓️ VARIABLES TEMPORALES
        fecha_col = next((c for c in df.columns if "fecha" in c.lower()), None)
        if fecha_col:
            df["año"] = df[fecha_col].dt.year
            df["mes"] = df[fecha_col].dt.month
            df["mes_texto"] = df[fecha_col].dt.strftime("%b")
            df["trimestre"] = df[fecha_col].dt.to_period("Q").astype(str)

        # === 🧾 MÉTRICAS DE VENTA POR CLIENTE ===
        if "id_cliente" in df.columns and "importe_total" in df.columns:
            resumen_clientes = df.groupby("id_cliente")["importe_total"].sum().rename("total_cliente")
            df = df.merge(resumen_clientes, on="id_cliente", how="left")
            df["participacion_cliente_%"] = round((df["importe_total"] / df["total_cliente"]) * 100, 2)

        # === 🏷️ MÉTRICAS DE CATEGORÍA ===
        if "categoria" in df.columns and "importe_total" in df.columns:
            resumen_categorias = df.groupby("categoria")["importe_total"].sum().rename("total_categoria")
            total_global = resumen_categorias.sum()
            df = df.merge(resumen_categorias, on="categoria", how="left")
            df["participacion_categoria_%"] = round((df["total_categoria"] / total_global) * 100, 2)

        # === 📦 MÉTRICAS DE PRODUCTO ===
        if "nombre_producto" in df.columns and "importe_total" in df.columns:
            resumen_productos = df.groupby("nombre_producto")["importe_total"].sum().rename("total_producto")
            df = df.merge(resumen_productos, on="nombre_producto", how="left")

            # Identificación de baja rotación
            umbral_baja_rotacion = df["total_producto"].quantile(0.25)
            df["baja_rotacion"] = np.where(df["total_producto"] <= umbral_baja_rotacion, True, False)

        # === 💰 TICKET PROMEDIO Y VENTA MENSUAL ===
        if "id_venta" in df.columns and "importe_total" in df.columns:
            ticket_venta = df.groupby("id_venta")["importe_total"].sum().rename("ticket_venta")
            df = df.merge(ticket_venta, on="id_venta", how="left")

        if {"año", "mes", "importe_total"}.issubset(df.columns):
            resumen_mensual = (
                df.groupby(["año", "mes"])["importe_total"]
                .agg(["sum", "mean"])
                .rename(columns={"sum": "total_mensual", "mean": "ticket_promedio_mensual"})
                .reset_index()
            )
            df = df.merge(resumen_mensual, on=["año", "mes"], how="left")

        # === 📊 RANKINGS ===
        if "categoria" in df.columns and "total_categoria" in df.columns:
            df["ranking_categoria"] = df.groupby("año")["total_categoria"].rank(ascending=False, method="dense").astype(int)

        if "nombre_producto" in df.columns and "total_producto" in df.columns:
            df["ranking_producto"] = df.groupby("año")["total_producto"].rank(ascending=False, method="dense").astype(int)

        if mostrar_mensajes:
            st.info("📈 Enriquecimiento completado con métricas gerenciales y de rotación.")

        return df

    except Exception as e:
        if mostrar_mensajes:
            st.warning(f"⚠️ Error durante el enriquecimiento: {e}")
        return df
