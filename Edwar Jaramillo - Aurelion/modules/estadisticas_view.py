import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from modules.utils.data_master import construir_tabla_maestra

sns.set(style="whitegrid")

# ============================================================
# FUNCIONES DE INTERPRETACIÓN AUTOMÁTICA
# ============================================================

def interpretar_distribucion(df, col):
    """Genera interpretación sobre la distribución de una variable numérica."""
    mean = df[col].mean()
    median = df[col].median()
    std = df[col].std()
    cv = (std / mean) * 100 if mean != 0 else np.nan
    skew = df[col].skew()

    interpretacion = []

    # Sesgo de la distribución
    if skew > 0.5:
        interpretacion.append("La distribución está sesgada a la derecha (asimetría positiva), con más valores bajos que altos.")
    elif skew < -0.5:
        interpretacion.append("La distribución está sesgada a la izquierda (asimetría negativa), con más valores altos que bajos.")
    else:
        interpretacion.append("La distribución es aproximadamente simétrica.")

    # Dispersión
    if cv < 20:
        interpretacion.append("Los valores son consistentes y presentan baja variabilidad.")
    elif cv < 50:
        interpretacion.append("Los valores muestran una variabilidad moderada.")
    else:
        interpretacion.append("Los valores presentan alta dispersión, indicando posibles subgrupos o datos heterogéneos.")

    return " ".join(interpretacion)


def interpretar_correlacion(corr):
    """Genera una interpretación automática del coeficiente de correlación."""
    if corr > 0.7:
        return "Existe una correlación **positiva fuerte**, es decir, cuando una variable aumenta, la otra también lo hace significativamente."
    elif corr > 0.4:
        return "Existe una **correlación positiva moderada**, las variables tienden a crecer juntas con cierta consistencia."
    elif corr > 0.1:
        return "Existe una **correlación positiva débil**, la relación es leve pero podría tener sentido en algunos casos."
    elif corr < -0.7:
        return "Existe una **correlación negativa fuerte**, cuando una variable aumenta, la otra tiende a disminuir significativamente."
    elif corr < -0.4:
        return "Existe una **correlación negativa moderada**, hay una tendencia inversa entre las variables."
    elif corr < -0.1:
        return "Existe una **correlación negativa débil**, la relación inversa es leve."
    else:
        return "No hay una correlación lineal significativa entre las variables."


def interpretar_confiabilidad(cv):
    """Evalúa la consistencia de los datos según el coeficiente de variación."""
    if cv < 15:
        return "Los datos son muy consistentes, con mínima variabilidad. Ideal para análisis predictivos."
    elif cv < 30:
        return "Los datos son bastante estables y confiables."
    elif cv < 50:
        return "Los datos tienen variabilidad moderada; puede haber diferencias notables entre grupos."
    else:
        return "Alta variabilidad: los datos son dispersos y menos confiables para predicciones directas."


# ============================================================
# FUNCIONES DE ANÁLISIS Y VISUALIZACIÓN
# ============================================================

def mostrar_estadistica_descriptiva(df):
    st.subheader("📊 Estadística descriptiva general")

    if df.empty:
        st.warning("⚠️ No hay datos disponibles en esta tabla.")
        return

    st.dataframe(df.describe().T)

    col = st.selectbox("Selecciona columna numérica:", df.select_dtypes(include=[np.number]).columns,key="desc_col")
    mean = df[col].mean()
    median = df[col].median()
    moda = df[col].mode().iloc[0]
    std = df[col].std()

    st.write(f"**Media:** {mean:.2f}")
    st.write(f"**Mediana:** {median:.2f}")
    st.write(f"**Moda:** {moda:.2f}")
    st.write(f"**Desviación estándar:** {std:.2f}")

    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, bins=20, ax=ax)
    ax.set_title(f"Distribución de {col}")
    st.pyplot(fig)

    # 🧠 Interpretación automática
    interpretacion = interpretar_distribucion(df, col)
    st.markdown(f"🧠 **Interpretación automática:** {interpretacion}")


def mostrar_medidas_posicion(df):
    st.subheader("📐 Medidas de posición")

    col = st.selectbox("Selecciona columna numérica:", df.select_dtypes(include=[np.number]).columns, key="posicion_col")

    minimo = df[col].min()
    maximo = df[col].max()
    q1, q2, q3 = df[col].quantile([0.25, 0.5, 0.75])
    rango = maximo - minimo

    st.write(f"**Mínimo:** {minimo:.2f}")
    st.write(f"**Máximo:** {maximo:.2f}")
    st.write(f"**Cuartiles:** Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}")
    st.write(f"**Rango:** {rango:.2f}")

    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)

    st.markdown("🧠 **Interpretación automática:** Los cuartiles indican la concentración de datos. Una caja compacta sugiere baja dispersión; una amplia, mayor variabilidad.")


def mostrar_correlaciones(df):
    st.subheader("🔗 Correlaciones entre variables")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        st.warning("⚠️ Se necesitan al menos dos columnas numéricas.")
        return

    x_var = st.selectbox("Variable 1 (X):", numeric_cols,key="corr_x")
    y_var = st.selectbox("Variable 2 (Y):", numeric_cols, index=1, key="corr_y")

    corr = df[[x_var, y_var]].corr().iloc[0, 1]
    st.write(f"**Coeficiente de correlación (r):** {corr:.3f}")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
    ax.set_title(f"Dispersión entre {x_var} y {y_var}")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0, ax=ax2)
    st.pyplot(fig2)

    # 🧠 Interpretación automática
    st.markdown(f"🧠 **Interpretación automática:** {interpretar_correlacion(corr)}")


def mostrar_confiabilidad(df):
    st.subheader("🧭 Evaluación de confiabilidad")

    col = st.selectbox("Selecciona una columna numérica:", df.select_dtypes(include=[np.number]).columns,key="conf_col")
    std = df[col].std()
    mean = df[col].mean()
    cv = (std / mean) * 100 if mean != 0 else np.nan

    st.write(f"**Desviación estándar:** {std:.2f}")
    st.write(f"**Coeficiente de variación (CV):** {cv:.2f}%")

    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, bins=20, ax=ax)
    ax.set_title(f"Distribución de {col}")
    st.pyplot(fig)

    st.markdown(f"🧠 **Interpretación automática:** {interpretar_confiabilidad(cv)}")


def mostrar_visualizaciones(df):
    st.subheader("📉 Visualizaciones estadísticas")

    tipo = st.selectbox("Selecciona tipo de gráfico:", ["Boxplot", "Heatmap", "Violinplot", "Histograma"],key="vis_tipo")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    
    if df.empty or len(numeric_cols) == 0:
        st.warning("⚠️ No hay columnas numéricas disponibles para graficar.")
        return
    
    # --------------------------------------------------
    # OPCIONES DE CONFIGURACIÓN GENERAL
    # --------------------------------------------------
    st.markdown("### ⚙️ Opciones de visualización")
    rotar_labels = st.checkbox("Rotar etiquetas del eje X", value=True)
    ajustar_ancho = st.slider("Ajustar ancho del gráfico:", 6, 20, 10)
    ordenar_por_media = st.checkbox("Ordenar categorías por valor promedio (solo aplica a boxplot/violinplot)", value=False)
    
    fig, ax = plt.subplots(figsize=(ajustar_ancho, 6))

    # --------------------------------------------------
    # 📦 BOX PLOT
    # --------------------------------------------------
    if tipo == "Boxplot":
        x = st.selectbox("Eje X (categoría):", cat_cols, key="vis_x_box")
        y = st.selectbox("Eje Y (numérico):", numeric_cols, key="vis_y_box")

        data_plot = df.copy()
        if ordenar_por_media and x in cat_cols:
            orden = data_plot.groupby(x)[y].mean().sort_values().index
        else:
            orden = None

        sns.boxplot(data=data_plot, x=x, y=y, order=orden, ax=ax)
        if rotar_labels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        st.markdown("🧠 **Interpretación:** El boxplot permite identificar asimetrías, concentraciones y outliers por categoría.")

    # --------------------------------------------------
    # 🎻 VIOLIN PLOT
    # --------------------------------------------------
    elif tipo == "Violinplot":
        x = st.selectbox("Eje X (categoría):", cat_cols, key="vis_x_violin")
        y = st.selectbox("Eje Y (numérico):", numeric_cols, key="vis_y_violin")

        data_plot = df.copy()
        if ordenar_por_media and x in cat_cols:
            orden = data_plot.groupby(x)[y].mean().sort_values().index
        else:
            orden = None

        sns.violinplot(data=data_plot, x=x, y=y, order=orden, ax=ax)
        if rotar_labels:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        st.markdown("🧠 **Interpretación:** El violinplot combina boxplot y densidad, mostrando la forma completa de la distribución.")

    # --------------------------------------------------
    # 🌡️ HEATMAP
    # --------------------------------------------------
    elif tipo == "Heatmap":
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        st.markdown("🧠 **Interpretación:** El mapa de calor muestra relaciones entre variables numéricas. Tonos rojos indican correlaciones positivas, azules negativas.")

    # --------------------------------------------------
    # 📊 HISTOGRAMA
    # --------------------------------------------------
    elif tipo == "Histograma":
        col = st.selectbox("Selecciona columna numérica:", numeric_cols, key="vis_hist_col")
        bins = st.slider("Número de intervalos (bins):", 5, 100, 20)
        kde = st.checkbox("Mostrar curva de densidad (KDE)", value=True)

        sns.histplot(df[col], bins=bins, kde=kde, color="steelblue", ax=ax)
        ax.set_title(f"Distribución de {col}", fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")

        st.markdown("🧠 **Interpretación:** El histograma muestra la frecuencia de los valores. Permite identificar concentración, sesgo y posibles outliers en los datos.")

    # --------------------------------------------------
    # MOSTRAR RESULTADO FINAL
    # --------------------------------------------------
    st.pyplot(fig)

def mostrar_analisis_gerencial(df):
    """
    Genera tres gráficos automáticos (ventas por categoría, evolución mensual y correlaciones)
    junto con una interpretación automática orientada a la gerencia.
    """

    st.subheader("📊 Análisis automático e interpretación gerencial")
    st.markdown("Este panel resume hallazgos clave del comportamiento de ventas del Minimarket Aurelion durante 2024.")

    # Validaciones iniciales
    if df.empty or "importe_total" not in df.columns:
        st.warning("⚠️ No hay datos válidos para generar el análisis.")
        return

    # ===============================
    # 1️⃣ VENTAS POR CATEGORÍA
    # ===============================
    st.markdown("### 🏷️ Ventas totales por categoría")

    if "categoria" in df.columns:
        ventas_cat = df.groupby("categoria")["importe_total"].sum().sort_values(ascending=False)
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(x=ventas_cat.values, y=ventas_cat.index, palette="viridis", ax=ax1)
        ax1.set_title("Ventas por Categoría")
        ax1.set_xlabel("Importe total")
        st.pyplot(fig1)

        top_cat = ventas_cat.idxmax()
        top_val = ventas_cat.max()
        bottom_cat = ventas_cat.idxmin()
        bottom_val = ventas_cat.min()
    else:
        top_cat = bottom_cat = top_val = bottom_val = None

    # ===============================
    # 2️⃣ EVOLUCIÓN MENSUAL DE VENTAS
    # ===============================
    st.markdown("### 📆 Evolución mensual de ventas (2024)")

    if {"mes", "importe_total"}.issubset(df.columns):
        ventas_mes = df.groupby("mes")["importe_total"].sum().sort_index()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.lineplot(x=ventas_mes.index, y=ventas_mes.values, marker="o", color="teal", ax=ax2)
        ax2.set_title("Evolución mensual de ventas")
        ax2.set_xlabel("Mes")
        ax2.set_ylabel("Importe total")
        st.pyplot(fig2)
    else:
        st.info("No hay información temporal disponible para graficar la evolución mensual.")

    # ===============================
    # 3️⃣ CORRELACIÓN ENTRE VARIABLES
    # ===============================
    st.markdown("### 🔗 Correlaciones principales")

    numeric_cols = df.select_dtypes(include=[np.number])
    if len(numeric_cols.columns) >= 3:
        corr = numeric_cols.corr(numeric_only=True)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax3)
        ax3.set_title("Matriz de correlaciones")
        st.pyplot(fig3)
    else:
        st.info("No hay suficientes variables numéricas para mostrar correlaciones.")

    # ===============================
    # 4️⃣ INTERPRETACIÓN AUTOMÁTICA
    # ===============================
    st.markdown("### 🧠 Interpretación gerencial automática")

    interpretaciones = []

    # a. Categorías dominantes
    if top_cat and bottom_cat:
        interpretaciones.append(
            f"La categoría **{top_cat}** concentra el mayor volumen de ventas "
            f"({top_val:,.0f} unidades monetarias), mientras que **{bottom_cat}** "
            f"presenta el menor desempeño ({bottom_val:,.0f})."
        )

    # b. Estacionalidad o crecimiento
    if "mes" in df.columns:
        mes_max = df.groupby("mes")["importe_total"].sum().idxmax()
        mes_min = df.groupby("mes")["importe_total"].sum().idxmin()
        interpretaciones.append(
            f"El mes con mayores ventas fue **{mes_max}**, mientras que el más bajo fue **{mes_min}**. "
            "Esto sugiere una estacionalidad en la demanda que puede aprovecharse para promociones o control de stock."
        )

    # c. Productos con baja rotación
    if "baja_rotacion" in df.columns:
        bajos = df[df["baja_rotacion"] == True]["nombre_producto"].nunique()
        total_prod = df["nombre_producto"].nunique()
        ratio = (bajos / total_prod) * 100 if total_prod else 0
        interpretaciones.append(
            f"Se detectaron **{bajos} productos ({ratio:.1f}% del total)** con baja rotación. "
            "Se recomienda revisar su demanda y considerar estrategias de liquidación o sustitución."
        )

    # d. Recomendaciones finales
    interpretaciones.append(
        "En general, se sugiere **ajustar el inventario mensual** en función de la estacionalidad "
        "y **enfocar promociones en las categorías de menor participación** para mejorar el equilibrio de ventas."
    )

    # Mostrar interpretaciones
    for texto in interpretaciones:
        st.markdown(f"🟣 {texto}")

    # ===============================
    # 5️⃣ CONCLUSIÓN GLOBAL
    # ===============================
    st.divider()
    st.markdown("#### 💡 Conclusión general")
    st.info(
        "El análisis muestra una estructura de ventas concentrada en pocas categorías con "
        "potencial de optimización. Se recomienda mantener seguimiento mensual, identificar "
        "clientes de alto valor y ajustar precios en productos de baja rotación para maximizar la rentabilidad."
    )


def mostrar_pareto_utilidad(df):
    """
    Analiza utilidad estimada y cantidad vendida por categoría o producto.
    Incluye filtros internos y dos modos de porcentaje (acumulado / sobre ventas totales).
    """

    st.subheader("📊 Pareto de utilidad y cantidad vendida")

    if df.empty or not {"precio_unitario", "cantidad"}.issubset(df.columns):
        st.warning("⚠️ No se encontraron columnas 'precio_unitario' o 'cantidad' necesarias para el análisis.")
        return

    df = df.copy()

    # ==========================================================
    # 1️⃣ FILTROS INTERNOS DE ANÁLISIS
    # ==========================================================
    st.markdown("### 🎚️ Filtros de análisis")

    col1, col2, col3 = st.columns(3)

    # --- Filtro por año ---
    if "año" in df.columns:
        años = sorted(df["año"].dropna().unique().tolist())
        filtro_año = col1.multiselect(
            "Filtrar por año:",
            ["(Todos)"] + años,
            default="(Todos)",
            key="pareto_filtro_año"
        )
        if "(Todos)" not in filtro_año:
            df = df[df["año"].isin(filtro_año)]

    # --- Filtro por mes ---
    if "mes" in df.columns:
        meses = sorted(df["mes"].dropna().unique().tolist())
        filtro_mes = col2.multiselect(
            "Filtrar por mes:",
            ["(Todos)"] + list(map(str, meses)),
            default="(Todos)",
            key="pareto_filtro_mes"
        )
        if "(Todos)" not in filtro_mes:
            df = df[df["mes"].astype(str).isin(filtro_mes)]

    # --- Filtro por categoría ---
    if "categoria" in df.columns:
        categorias = sorted(df["categoria"].dropna().unique().tolist())
        filtro_cat = col3.multiselect(
            "Filtrar por categoría:",
            ["(Todas)"] + categorias,
            default="(Todas)",
            key="pareto_filtro_categoria"
        )
        if "(Todas)" not in filtro_cat:
            df = df[df["categoria"].isin(filtro_cat)]

    if df.empty:
        st.warning("⚠️ No hay datos después de aplicar los filtros seleccionados.")
        return

    st.divider()

    # ==========================================================
    # 2️⃣ CONFIGURACIONES DEL ANÁLISIS
    # ==========================================================
    colA, colB = st.columns(2)

    nivel = colA.radio(
        "Nivel de análisis:",
        ["Por categoría", "Por producto"],
        horizontal=True,
        key="pareto_nivel"
    )

    modo_porcentaje = colB.radio(
        "Modo de porcentaje:",
        ["% acumulado (Pareto clásico)", "% sobre ventas totales"],
        horizontal=True,
        key="pareto_modo"
    )

    # ==========================================================
    # 3️⃣ CÁLCULOS DE UTILIDAD E INGRESO
    # ==========================================================
    df["ingreso"] = df["precio_unitario"] * df["cantidad"]

    # Márgenes promedio estimados según categoría
    margenes = {
        "Alimentos": 0.05,
        "Bebidas": 0.18,
        "Bebidas alcohólicas": 0.12,
        "Limpieza": 0.20,
        "Lácteos": 0.10,
        "Panadería": 0.15,
        "Cuidado personal": 0.15,
        "Dulces": 0.15,
        "Snacks y Dulces": 0.15,
        "Granos y Cereales": 0.05,
        "Verduras": 0.05
    }

    def utilidad_estim(row):
        categoria = str(row.get("categoria", "Otros")).strip()
        margen = margenes.get(categoria, 0.10)  # margen default 10%
        return row["ingreso"] * margen

    df["utilidad_estimada"] = df.apply(utilidad_estim, axis=1)

    # ==========================================================
    # 4️⃣ AGRUPAR SEGÚN NIVEL DE ANÁLISIS
    # ==========================================================
    if nivel == "Por categoría" and "categoria" in df.columns:
        agrupado = (
            df.groupby("categoria")
            .agg({
                "cantidad": "sum",
                "ingreso": "sum",
                "utilidad_estimada": "sum"
            })
            .sort_values("utilidad_estimada", ascending=False)
            .reset_index()
        )
        nombre_col = "categoria"
    else:
        agrupado = (
            df.groupby("nombre_producto")
            .agg({
                "cantidad": "sum",
                "ingreso": "sum",
                "utilidad_estimada": "sum"
            })
            .sort_values("utilidad_estimada", ascending=False)
            .reset_index()
        )
        nombre_col = "nombre_producto"

    # ==========================================================
    # 5️⃣ CÁLCULO DE PORCENTAJES SEGÚN EL MODO
    # ==========================================================
    if modo_porcentaje == "% acumulado (Pareto clásico)":
        agrupado["%"] = (agrupado["utilidad_estimada"].cumsum() / agrupado["utilidad_estimada"].sum()) * 100
    else:
        total_ventas = agrupado["ingreso"].sum()
        agrupado["%"] = (agrupado["utilidad_estimada"] / total_ventas) * 100

    agrupado.rename(
        columns={
            "%": "% Acumulado" if modo_porcentaje.startswith("% acumulado") else "% sobre ventas"
        },
        inplace=True
    )

    # ==========================================================
    # 6️⃣ GRÁFICO PARETO INTERACTIVO
    # ==========================================================
    fig = go.Figure()

    # Barras de utilidad
    fig.add_trace(go.Bar(
        x=agrupado[nombre_col],
        y=agrupado["utilidad_estimada"],
        name="Utilidad estimada ($)",
        marker_color="royalblue",
        yaxis="y1"
    ))

    # Línea de porcentaje
    col_pct = agrupado.columns[-1]
    fig.add_trace(go.Scatter(
        x=agrupado[nombre_col],
        y=agrupado[col_pct],
        name=col_pct,
        mode="lines+markers",
        marker=dict(color="darkorange"),
        yaxis="y2"
    ))

    fig.update_layout(
        title=f"📈 Gráfico Pareto - {nivel} ({col_pct})",
        xaxis=dict(title=nombre_col.capitalize(), tickangle=45),
        yaxis=dict(title="Utilidad estimada ($)"),
        yaxis2=dict(
            title=col_pct,
            overlaying="y",
            side="right",
            range=[0, 110 if 'acumulado' in col_pct.lower() else max(agrupado[col_pct]) * 1.2]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        height=600,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # 7️⃣ TABLA RESUMEN E INSIGHT AUTOMÁTICO
    # ==========================================================
    st.markdown("### 📋 Resumen de utilidades")
    st.dataframe(agrupado.head(20), use_container_width=True)

    top_item = agrupado.iloc[0, 0]
    top_util = agrupado["utilidad_estimada"].iloc[0]
    pct = agrupado.iloc[0, -1]

    st.markdown(
        f"💡 **Insight:** El elemento **{top_item}** concentra la mayor utilidad estimada "
        f"(${top_util:,.0f}), representando el **{pct:.1f}%** {col_pct.lower()}."
    )

    st.caption(
        "🔍 Este gráfico permite visualizar cómo pocas categorías o productos concentran la mayoría de la utilidad total "
        "(principio de Pareto 80/20), o qué porcentaje representa cada uno sobre el total de ventas."
    )


def mostrar_rentabilidad_productos(df):
    """
    Analiza la rentabilidad, margen y ROI por producto,
    permitiendo ajustar manualmente el porcentaje de margen
    y aplicar filtros por categoría y mes si están disponibles.
    """
    st.subheader("💰 Utilidad y ROI por producto")

    if df.empty or "precio_unitario" not in df.columns or "cantidad" not in df.columns:
        st.warning("⚠️ No se encontraron columnas 'precio_unitario' o 'cantidad' para calcular rentabilidad.")
        return

    df = df.copy()

    # --------------------------------------------------------
    # 1️⃣ Filtros dinámicos de categoría y mes
    # --------------------------------------------------------
    st.markdown("### 🎚️ Filtros de análisis")

    col1, col2 = st.columns(2)
    filtro_categoria, filtro_mes = None, None

    if "categoria" in df.columns:
        categorias = sorted(df["categoria"].dropna().unique().tolist())
        filtro_categoria = col1.multiselect("Filtrar por categoría:", ["(Todas)"] + categorias, default="(Todas)")

    if "mes" in df.columns:
        meses = sorted(df["mes"].dropna().unique().tolist())
        filtro_mes = col2.multiselect("Filtrar por mes:", ["(Todos)"] + list(map(str, meses)), default="(Todos)")

    # Aplicar filtros
    if filtro_categoria and "(Todas)" not in filtro_categoria:
        df = df[df["categoria"].isin(filtro_categoria)]
    if filtro_mes and "(Todos)" not in filtro_mes:
        df = df[df["mes"].astype(str).isin(filtro_mes)]

    if df.empty:
        st.warning("⚠️ No hay datos después de aplicar los filtros seleccionados.")
        return

    # --------------------------------------------------------
    # 2️⃣ Margen editable
    # --------------------------------------------------------
    st.markdown("### ⚙️ Configuración de margen de ganancia")
    margen_input = st.number_input(
        "Margen de ganancia (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
        help="Porcentaje de margen sobre el precio unitario (por defecto 30 %)."
    )
    margen_factor = (100 - margen_input) / 100  # Ejemplo: 30 % → costo = 0.7 * precio

    # --------------------------------------------------------
    # 3️⃣ Cálculos de utilidad
    # --------------------------------------------------------
    df["costo_unitario"] = df["precio_unitario"] * margen_factor
    df["ganancia_unitaria"] = df["precio_unitario"] - df["costo_unitario"]
    df["ganancia_total"] = df["ganancia_unitaria"] * df["cantidad"]
    df["margen_%"] = (df["ganancia_unitaria"] / df["precio_unitario"]) * 100
    df["ROI_%"] = (df["ganancia_total"] / (df["costo_unitario"] * df["cantidad"])) * 100

    # --------------------------------------------------------
    # 4️⃣ Agrupación por producto
    # --------------------------------------------------------
    rentabilidad = (
        df.groupby("nombre_producto")
        .agg({
            "categoria": "first" if "categoria" in df.columns else lambda x: None,
            "cantidad": "sum",
            "importe_total": "sum",
            "ganancia_total": "sum",
            "margen_%": "mean",
            "ROI_%": "mean"
        })
        .sort_values("ganancia_total", ascending=False)
        .reset_index()
    )

    st.markdown("### 🧾 Tabla resumen de utilidad por producto")
    st.dataframe(rentabilidad.head(20), use_container_width=True)

    # --------------------------------------------------------
    # 5️⃣ Gráficos de rentabilidad
    # --------------------------------------------------------
    top = rentabilidad.head(10)
    bottom = rentabilidad.tail(10)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=top, x="ganancia_total", y="nombre_producto", ax=ax[0], palette="Greens_r")
    sns.barplot(data=bottom, x="ganancia_total", y="nombre_producto", ax=ax[1], palette="Reds_r")

    ax[0].set_title("🔝 Productos más rentables")
    ax[1].set_title("⚠️ Productos menos rentables")
    for a in ax:
        a.set_xlabel("Ganancia total ($)")
        a.set_ylabel("")
    st.pyplot(fig)

    # --------------------------------------------------------
    # 6️⃣ Métricas e interpretación automática
    # --------------------------------------------------------
    margen_prom = rentabilidad["margen_%"].mean()
    roi_prom = rentabilidad["ROI_%"].mean()

    st.markdown("### 🧠 Interpretación automática")
    st.write(f"📈 **Margen promedio:** {margen_prom:.2f}%")
    st.write(f"💹 **ROI promedio:** {roi_prom:.2f}%")

    if roi_prom > 40:
        st.success("Excelente nivel de rentabilidad general. El mix de productos genera retornos altos sobre la inversión.")
    elif roi_prom > 20:
        st.info("Buen desempeño general, aunque algunos productos podrían optimizar precios o costos.")
    else:
        st.warning("Rentabilidad baja: se recomienda revisar estructura de costos o estrategias de precios.")

    st.caption("🔄 Los resultados se actualizan automáticamente al cambiar filtros o margen de ganancia.")


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def mostrar_estadisticas(datasets):
    """Vista principal del módulo de análisis y visualización."""
    st.title("📈 Análisis Estadístico y Visualización de Datos")

    if not datasets:
        st.warning("⚠️ No hay datasets cargados en memoria.")
        return

    # ===============================================================
    # Construir y agregar la tabla maestra al selector
    # ===============================================================
    tabla_maestra = construir_tabla_maestra(datasets, mostrar_mensajes=False)

    # Crear copia del diccionario de datasets y añadir la tabla maestra si existe
    datasets_para_analisis = dict(datasets)
    if not tabla_maestra.empty:
        datasets_para_analisis["tabla_maestra"] = tabla_maestra

    # ===============================================================
    # Selector de tabla para análisis
    # ===============================================================
    tabla_seleccionada = st.selectbox(
        "Selecciona la tabla para analizar:",
        list(datasets_para_analisis.keys())
    )
    df = datasets_para_analisis[tabla_seleccionada]

    st.markdown(f"### Analizando tabla: `{tabla_seleccionada}`")

    # ===============================================================
    # Pestañas de análisis
    # ===============================================================
    tabs = st.tabs([
        "Estadística descriptiva",
        "Medidas de posición",
        "Correlaciones",
        "Confiabilidad",
        "Visualizaciones",
        "📊 Análisis automático\n\ninterpretación gerencial",
        "💰 Utilidad y ROI",
        "📈 Pareto de utilidad"
    ])

    with tabs[0]:
        mostrar_estadistica_descriptiva(df)
    with tabs[1]:
        mostrar_medidas_posicion(df)
    with tabs[2]:
        mostrar_correlaciones(df)
    with tabs[3]:
        mostrar_confiabilidad(df)
    with tabs[4]:
        mostrar_visualizaciones(df)
    with tabs[5]:
        mostrar_analisis_gerencial(df)
    with tabs[6]:
        mostrar_rentabilidad_productos(df)
    with tabs[7]:
        mostrar_pareto_utilidad(df)