# modules/reportes_view.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import io
from datetime import datetime
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
        "Correlaciones globales",
        "📊 Análisis\n\ngerencial avanzado"
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

        # Calcular matriz de correlaciones
        corr = maestra.corr(numeric_only=True)

        # --- Limpiar variables no informativas ---
        corr = corr.dropna(how="all", axis=0).dropna(how="all", axis=1)
        if corr.empty:
            st.warning("⚠️ No hay variables numéricas suficientes para calcular correlaciones.")
        else:
            # --- Filtro por umbral de correlación ---
            umbral = st.slider("Umbral mínimo de correlación a mostrar", 0.0, 1.0, 0.6, 0.05)
            mask = corr.abs() >= umbral
            corr_filtrado = corr.where(mask)

            # --- Heatmap interactivo con Plotly ---
            fig = px.imshow(
                corr_filtrado,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                origin="lower",
                zmin=-1,
                zmax=1,
                labels=dict(color="Coeficiente de correlación"),
                title=f"Matriz de correlación (|r| ≥ {umbral})"
            )
            fig.update_layout(
                width=950,
                height=700,
                margin=dict(l=60, r=30, t=50, b=30),
                coloraxis_colorbar=dict(title="Correlación", len=0.75),
                font=dict(size=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- Tabla de correlaciones significativas ---
            st.markdown("### 🧮 Correlaciones más significativas")
            top_corrs = (
                corr.unstack()
                .reset_index()
                .rename(columns={"level_0": "Variable A", "level_1": "Variable B", 0: "Correlación"})
            )

            top_corrs = top_corrs[
                (top_corrs["Variable A"] != top_corrs["Variable B"]) &
                (abs(top_corrs["Correlación"]) >= umbral)
            ].sort_values("Correlación", ascending=False).drop_duplicates(subset=["Variable A", "Variable B"])

            if not top_corrs.empty:
                st.dataframe(top_corrs.head(15), use_container_width=True)
                st.info(
                    "📊 Los valores cercanos a **+1** indican relación directa fuerte (ambas aumentan juntas), "
                    "mientras que valores cercanos a **-1** indican relación inversa (una sube, la otra baja)."
                )
            else:
                st.info(f"✅ No se detectaron correlaciones fuertes con |r| ≥ {umbral}.")
    
    with tabs[5]:
        mostrar_analisis_gerencial(maestra)


def generar_reporte_pdf(interpretaciones, graficas, titulo="Reporte Gerencial - Minimarket Aurelion"):
    """
    Genera un archivo PDF con las gráficas e interpretaciones del análisis.
    Compatible con caracteres Unicode (emojis, acentos, etc.).
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Registrar fuente Unicode (asegúrate que la ruta exista)
    pdf.add_font("DejaVu", "", "assets/DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "assets/DejaVuSans-Bold.ttf", uni=True)
    pdf.set_font("DejaVu", "B", 16)

    # ======= Portada =======
    pdf.add_page()
    pdf.cell(0, 10, titulo, ln=True, align="C")
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=True, align="C")
    pdf.ln(15)
    pdf.multi_cell(0, 8, "📊 Este informe contiene los principales hallazgos y conclusiones gerenciales del análisis estadístico del minimarket Aurelion.")
    pdf.ln(10)

    # ======= Gráficas =======
    for idx, img in enumerate(graficas, start=1):
        pdf.add_page()
        pdf.set_font("DejaVu", "B", 13)
        pdf.cell(0, 10, f"Gráfica {idx}", ln=True)
        pdf.image(img, x=10, y=30, w=180)
        pdf.ln(120)

    # ======= Conclusiones =======
    pdf.add_page()
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "💡 Conclusiones y recomendaciones", ln=True)
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 11)
    for parrafo in interpretaciones:
        pdf.multi_cell(0, 8, parrafo)
        pdf.ln(5)

    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer


def mostrar_analisis_gerencial(maestra):
    """
    Panel de análisis gerencial avanzado e interactivo:
    - Filtros por año, categoría, mes y producto
    - Ventas por categoría y mes
    - Productos con baja rotación (ventas y cantidades)
    - Insights automáticos y conclusiones textuales
    """

    st.subheader("📊 Análisis Gerencial Avanzado e Interactivo")

    if maestra.empty:
        st.warning("⚠️ No hay información disponible para el análisis.")
        return

    # ======================================================
    # 🔍 FILTROS LATERALES
    # ======================================================
    st.sidebar.markdown("### 🔎 Filtros del análisis")

    # Año
    if "año" in maestra.columns:
        anios = sorted(maestra["año"].dropna().unique())
        anio_sel = st.sidebar.multiselect("Año:", anios, default=anios)
        maestra = maestra[maestra["año"].isin(anio_sel)]

    # Categoría
    if "categoria" in maestra.columns:
        categorias = sorted(maestra["categoria"].dropna().unique())
        cat_sel = st.sidebar.multiselect("Categorías:", categorias, default=categorias)
        maestra = maestra[maestra["categoria"].isin(cat_sel)]

    # Mes
    if "mes" in maestra.columns:
        meses = sorted(maestra["mes"].dropna().unique())
        mes_sel = st.sidebar.multiselect("Mes:", meses, default=meses)
        maestra = maestra[maestra["mes"].isin(mes_sel)]

    # Producto
    if "nombre_producto" in maestra.columns:
        productos = sorted(maestra["nombre_producto"].dropna().unique())
        prod_sel = st.sidebar.multiselect("Productos (opcional):", productos)
        if prod_sel:
            maestra = maestra[maestra["nombre_producto"].isin(prod_sel)]

    st.markdown(
        f"📁 **Datos filtrados:** {len(maestra)} registros · "
        f"{maestra['nombre_producto'].nunique() if 'nombre_producto' in maestra.columns else 0} productos · "
        f"{maestra['categoria'].nunique() if 'categoria' in maestra.columns else 0} categorías"
    )

    st.divider()

    # ======================================================
    # 1️⃣ VENTAS POR CATEGORÍA Y MES (INTERACTIVO)
    # ======================================================
    st.markdown("### 🏷️ Ventas por categoría y mes")

    if {"categoria", "mes", "importe_total"}.issubset(maestra.columns):
        ventas_cat_mes = (
            maestra.groupby(["mes", "categoria"], as_index=False)
            .agg({"importe_total": "sum"})
        )

        fig1 = px.line(
            ventas_cat_mes,
            x="mes",
            y="importe_total",
            color="categoria",
            markers=True,
            hover_name="categoria",
            hover_data={"importe_total": ":,.0f"},
            title="Evolución mensual de ventas por categoría"
        )
        fig1.update_layout(
            xaxis_title="Mes",
            yaxis_title="Ventas totales ($)",
            legend_title="Categoría",
            hovermode="x unified",
            width=950,
            height=500
        )
        st.plotly_chart(fig1, use_container_width=True)

        top_mes = ventas_cat_mes.groupby("mes")["importe_total"].sum().idxmax()
        st.markdown(f"🧠 **Insight:** El mes con mayores ventas totales fue **{top_mes}**, destacando las categorías líderes del período.")
    else:
        st.info("⚠️ No se dispone de columnas 'categoria' y 'mes' para generar esta vista.")

    st.divider()

    # ======================================================
    # 2️⃣ PRODUCTOS CON BAJA ROTACIÓN Y EVOLUCIÓN MENSUAL
    # ======================================================
    st.markdown("### 🧾 Productos con baja rotación y evolución de ventas")

    if {"baja_rotacion", "mes", "nombre_producto"}.issubset(maestra.columns):
        baja_rot = maestra[maestra["baja_rotacion"] == True]
        if baja_rot.empty:
            st.success("✅ No se encontraron productos con baja rotación en el período analizado.")
        else:
            ventas_baja_mes = (
                baja_rot.groupby(["mes", "nombre_producto"], as_index=False)
                .agg({"importe_total": "sum", "cantidad": "sum"})
            )

            fig2 = px.line(
                ventas_baja_mes,
                x="mes",
                y="importe_total",
                color="nombre_producto",
                markers=True,
                hover_name="nombre_producto",
                hover_data={"importe_total": ":,.0f", "cantidad": ":,.0f"},
                title="Evolución mensual de ventas - Productos de baja rotación"
            )
            fig2.update_layout(
                xaxis_title="Mes",
                yaxis_title="Ventas totales ($)",
                legend_title="Producto",
                hovermode="x unified",
                width=950,
                height=500
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown(
                f"🧠 **Insight:** Se identificaron **{baja_rot['nombre_producto'].nunique()} productos** de baja rotación. "
                "Analiza estrategias de promoción o sustitución para los menos vendidos."
            )
    else:
        st.info("⚠️ No hay información de rotación o temporalidad para analizar productos de baja venta.")

    st.divider()

    # ======================================================
    # 3️⃣ CONCLUSIONES AUTOMÁTICAS Y RECOMENDACIONES
    # ======================================================
    st.markdown("### 💡 Conclusiones y recomendaciones automáticas")

    total_ventas = maestra["importe_total"].sum()
    total_baja = maestra.get("baja_rotacion", pd.Series([False]*len(maestra))).sum()
    total_prod = maestra["nombre_producto"].nunique() if "nombre_producto" in maestra.columns else 0
    ratio_baja = (total_baja / total_prod) * 100 if total_prod > 0 else 0

    principales_cats = (
        maestra.groupby("categoria")["importe_total"]
        .sum()
        .sort_values(ascending=False)
        .head(3)
    )

    # === Mantener tus conclusiones originales ===
    interpretacion = [
        f"📈 El volumen total de ventas asciende a **${total_ventas:,.0f}**, con mayor contribución de las categorías: "
        + ", ".join(principales_cats.index.tolist()) + ".",
        f"📉 Se identificó un **{ratio_baja:.1f}%** de productos con baja rotación. Se recomienda reforzar campañas de promoción o revisar precios.",
        "🕒 Las ventas muestran una tendencia mensual estable, con picos en algunos períodos estacionales. Ajustar stock e inventarios según el comportamiento histórico.",
        "💰 Priorizar productos líderes y optimizar el catálogo de baja rotación puede incrementar la rentabilidad general del minimarket."
    ]

    for txt in interpretacion:
        st.markdown(txt)

    st.success("✅ Reporte gerencial generado con éxito.")

        # ======================================================
    # 📤 DESCARGAR REPORTE PDF
    # ======================================================
    st.divider()
    st.markdown("### 📤 Exportar reporte")

    # --- Capturar gráficas (de Plotly) como imágenes ---
    graficas = []
    try:
        fig1_bytes = io.BytesIO(fig1.to_image(format="png"))
        graficas.append(fig1_bytes)
        if "fig2" in locals():
            fig2_bytes = io.BytesIO(fig2.to_image(format="png"))
            graficas.append(fig2_bytes)
    except Exception as e:
        st.warning(f"⚠️ No se pudieron convertir las gráficas a imagen: {e}")

    if st.button("📄 Descargar reporte en PDF"):
        buffer_pdf = generar_reporte_pdf(interpretacion, graficas)
        st.download_button(
            label="⬇️ Guardar archivo PDF",
            data=buffer_pdf,
            file_name=f"Reporte_Gerencial_Aurelion_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

