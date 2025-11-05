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

        # === 1️⃣ Validación
        if detalle.empty or ventas.empty or productos.empty or clientes.empty:
            if mostrar_mensajes:
                st.warning("⚠️ No se puede crear la tabla maestra: falta al menos una tabla base.")
            return pd.DataFrame()

        # === 2️⃣ Unión progresiva
        maestra = (
            detalle
            .merge(ventas, on="id_venta", how="left", suffixes=("", "_venta"))
            .merge(productos, on="id_producto", how="left", suffixes=("", "_producto"))
            .merge(clientes, on="id_cliente", how="left", suffixes=("", "_cliente"))
        )

        # === 3️⃣ Creación robusta de 'importe_total'
        posibles_precio = [c for c in maestra.columns if "precio" in c.lower()]
        posibles_cant = [c for c in maestra.columns if "cant" in c.lower()]

        if posibles_precio and posibles_cant:
            col_precio = posibles_precio[0]
            col_cant = posibles_cant[0]
            maestra["importe_total"] = maestra[col_precio] * maestra[col_cant]
        else:
            maestra["importe_total"] = np.nan
            if mostrar_mensajes:
                st.warning("⚠️ No se detectaron columnas 'precio' o 'cantidad' para calcular importe_total.")

        # === 4️⃣ Conversión de fechas
        posibles_fechas = [col for col in maestra.columns if "fecha" in col.lower()]
        for col in posibles_fechas:
            maestra[col] = pd.to_datetime(maestra[col], errors="coerce")

        # === 5️⃣ Enriquecimiento opcional
        if enriquecer:
            maestra = _enriquecer_tabla_maestra(maestra, mostrar_mensajes)

        # === 6️⃣ Limpieza final
        maestra = maestra.drop_duplicates()

        if mostrar_mensajes:
            st.success(f"✅ Tabla maestra creada correctamente ({len(maestra)} registros, {len(maestra.columns)} columnas).")

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
    Agrega métricas derivadas y columnas auxiliares automáticamente
    (por ejemplo: mes, año, ticket promedio, % por categoría).
    """

    try:
        # === 🗓️ VARIABLES TEMPORALES
        fecha_col = next((c for c in df.columns if "fecha" in c.lower()), None)
        if fecha_col:
            df["año"] = df[fecha_col].dt.year
            df["mes"] = df[fecha_col].dt.month
            df["mes_texto"] = df[fecha_col].dt.strftime("%b")
            df["trimestre"] = df[fecha_col].dt.to_period("Q").astype(str)

        # === 🛍️ MÉTRICAS DE CLIENTE
        if "id_cliente" in df.columns and "importe_total" in df.columns:
            resumen_clientes = (
                df.groupby("id_cliente")["importe_total"].sum().rename("total_cliente")
            )
            df = df.merge(resumen_clientes, on="id_cliente", how="left")
            df["participacion_cliente_%"] = round((df["importe_total"] / df["total_cliente"]) * 100, 2)

        # === 🏷️ MÉTRICAS DE CATEGORÍA
        if "categoria" in df.columns and "importe_total" in df.columns:
            resumen_categorias = (
                df.groupby("categoria")["importe_total"].sum().rename("total_categoria")
            )
            total_global = resumen_categorias.sum()
            df = df.merge(resumen_categorias, on="categoria", how="left")
            df["participacion_categoria_%"] = round((df["total_categoria"] / total_global) * 100, 2)

        # === 📦 MÉTRICAS DE PRODUCTO
        if "nombre_producto" in df.columns and "importe_total" in df.columns:
            resumen_productos = (
                df.groupby("nombre_producto")["importe_total"].sum().rename("total_producto")
            )
            df = df.merge(resumen_productos, on="nombre_producto", how="left")

        # === 🧾 MÉTRICAS DE VENTA
        if "id_venta" in df.columns and "importe_total" in df.columns:
            ticket_por_venta = (
                df.groupby("id_venta")["importe_total"].sum().rename("ticket_venta")
            )
            df = df.merge(ticket_por_venta, on="id_venta", how="left")

        if mostrar_mensajes:
            st.info("📈 Enriquecimiento automático completado con éxito (se agregaron métricas derivadas).")

        return df

    except Exception as e:
        if mostrar_mensajes:
            st.warning(f"⚠️ Error durante el enriquecimiento: {e}")
        return df
