import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
        "📊 Análisis automático\n\ninterpretación gerencial"
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