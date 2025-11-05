# modules/utils/data_master.py

import streamlit as st
import pandas as pd
import numpy as np

def construir_tabla_maestra(datasets, mostrar_mensajes=True):
    """
    Combina productos, clientes, ventas y detalle_ventas en una sola tabla analítica consolidada.
    Cumple con el principio de responsabilidad única (SRP) para uso transversal.

    Args:
        datasets (dict): Diccionario con DataFrames cargados en memoria.
        mostrar_mensajes (bool): Si True, muestra mensajes en Streamlit (para vistas interactivas).

    Returns:
        pd.DataFrame: DataFrame consolidado (tabla maestra) con métricas derivadas.
    """
    try:
        productos = datasets.get("productos", pd.DataFrame())
        clientes = datasets.get("clientes", pd.DataFrame())
        ventas = datasets.get("ventas", pd.DataFrame())
        detalle = datasets.get("detalle_ventas", pd.DataFrame())

        # Validación de existencia
        if detalle.empty or ventas.empty or productos.empty or clientes.empty:
            if mostrar_mensajes:
                st.warning("⚠️ No se puede crear la tabla maestra: falta al menos una tabla base.")
            return pd.DataFrame()

        # === 🔗 Combinación progresiva (respetando integridad referencial)
        maestra = (
            detalle
            .merge(ventas, on="id_venta", how="left", suffixes=("", "_venta"))
            .merge(productos, on="id_producto", how="left", suffixes=("", "_producto"))
            .merge(clientes, on="id_cliente", how="left", suffixes=("", "_cliente"))
        )

        # === 💰 Creación robusta de 'importe_total'
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

        # === 🗓️ Conversión de fechas
        posibles_fechas = [col for col in maestra.columns if "fecha" in col.lower()]
        for col in posibles_fechas:
            maestra[col] = pd.to_datetime(maestra[col], errors="coerce")

        # === 🧹 Limpieza de duplicados
        maestra = maestra.drop_duplicates()

        if mostrar_mensajes:
            st.info(f"✅ Tabla maestra creada correctamente ({len(maestra)} registros, {len(maestra.columns)} columnas).")

        return maestra

    except Exception as e:
        if mostrar_mensajes:
            st.error(f"❌ Error al construir la tabla maestra: {e}")
        return pd.DataFrame()
