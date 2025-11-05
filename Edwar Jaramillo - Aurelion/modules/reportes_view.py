# modules/reportes_view.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from modules.utils.data_master import construir_tabla_maestra

sns.set(style="whitegrid")

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def mostrar_reportes(datasets):
    """Dashboard ejecutivo de reportes y KPIs basado en la tabla maestra."""
    st.title("📊 Reportes Gerenciales y KPIs")

    # =======================================================
    # 🧩 1️⃣ Construcción de la tabla maestra analítica
    # =======================================================
    maestra = construir_tabla_maestra(datasets)
    if maestra.empty:
        return

    # =======================================================
    # 🧩 2️⃣ KPIs PRINCIPALES
    # =======================================================
    st.markdown("## 📈 Indicadores Clave (KPIs)")

    total_ventas = maestra["importe_total"].sum()
    total_clientes = maestra["id_cliente"].nunique()
    total_productos = maestra["id_producto"].nunique()
    promedio_ticket = maestra.groupby("id_venta")["importe_total"].sum().mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Ventas totales", f"${total_ventas:,.0f}")
    col2.metric("🧍‍♂️ Clientes únicos", total_clientes)
    col3.metric("📦 Productos vendidos", total_productos)
    col4.metric("🧾 Ticket promedio", f"${promedio_ticket:,.0f}")

    st.divider()

    # =======================================================
    # 🧩 3️⃣ ANÁLISIS VISUAL POR DIMENSIONES
    # =======================================================

    tabs = st.tabs([
        "Productos más vendidos",
        "Categorías más rentables",
        "Clientes principales",
        "Ventas por mes",
        "Correlaciones globales"
    ])

    # === Productos más vendidos ===
    with tabs[0]:
        st.subheader("🏆 Top productos por cantidad vendida")
        top_productos = (
            maestra.groupby("nombre_producto")["cantidad"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top_productos.values, y=top_productos.index, ax=ax)
        ax.set_xlabel("Cantidad total vendida")
        st.pyplot(fig)

        st.markdown(f"🧠 **Insight:** El producto más vendido es **{top_productos.index[0]}**, con {top_productos.iloc[0]} unidades.")

    # === Categorías más rentables ===
    with tabs[1]:
        st.subheader("💸 Categorías con mayor valor de ventas")
        if "categoria" in maestra.columns:
            top_categorias = (
                maestra.groupby("categoria")["importe_total"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=top_categorias.values, y=top_categorias.index, ax=ax)
            ax.set_xlabel("Importe total vendido")
            st.pyplot(fig)

            st.markdown(f"🧠 **Insight:** La categoría **{top_categorias.index[0]}** representa el mayor ingreso, con un total de ${top_categorias.iloc[0]:,.0f}.")
        else:
            st.info("⚠️ No hay columna 'categoria' disponible en la base de datos.")

    # === Clientes principales ===
    with tabs[2]:
        st.subheader("👥 Clientes con mayor volumen de compras")
        if "nombre_cliente" in maestra.columns:
            top_clientes = (
                maestra.groupby("nombre_cliente")["importe_total"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=top_clientes.values, y=top_clientes.index, ax=ax)
            ax.set_xlabel("Importe total comprado")
            st.pyplot(fig)

            st.markdown(f"🧠 **Insight:** El cliente **{top_clientes.index[0]}** lidera las compras con ${top_clientes.iloc[0]:,.0f} en ventas totales.")
        else:
            st.info("⚠️ No hay columna 'nombre_cliente' disponible en la base de datos.")

    # === Ventas por mes ===
    with tabs[3]:
        st.subheader("🗓️ Ventas totales por mes")
        fecha_col = next((c for c in maestra.columns if "fecha" in c.lower()), None)
        if fecha_col:
            maestra["mes"] = maestra[fecha_col].dt.to_period("M").astype(str)
            ventas_mes = maestra.groupby("mes")["importe_total"].sum()
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(x=ventas_mes.index, y=ventas_mes.values, marker="o", ax=ax)
            ax.set_xlabel("Mes")
            ax.set_ylabel("Ventas totales")
            st.pyplot(fig)

            mes_top = ventas_mes.idxmax()
            st.markdown(f"🧠 **Insight:** El mes con mayor facturación fue **{mes_top}**, con un total de ${ventas_mes.max():,.0f}.")
        else:
            st.info("⚠️ No se detectó ninguna columna de fecha para agrupar por mes.")

    # === Correlaciones globales ===
    with tabs[4]:
        st.subheader("📈 Correlaciones entre variables numéricas")
        corr = maestra.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

        st.markdown("🧠 **Insight:** Este mapa de calor muestra cómo se relacionan las métricas numéricas entre sí, permitiendo detectar factores que influyen en las ventas.")
