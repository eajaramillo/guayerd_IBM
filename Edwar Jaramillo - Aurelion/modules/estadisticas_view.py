import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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

    # ==============================
    # 🔹 CORRELACIÓN ENTRE DOS VARIABLES
    # ==============================
    st.markdown("### 🔸 Correlación entre dos variables específicas")
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variable X:", numeric_cols, key="corr_x")
    with col2:
        y_var = st.selectbox("Variable Y:", numeric_cols, index=1, key="corr_y")

    corr = df[[x_var, y_var]].corr().iloc[0, 1]
    st.write(f"**Coeficiente de correlación (r):** `{corr:.3f}`")

    # Gráfico de dispersión
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax, alpha=0.7)
    ax.set_title(f"Dispersión entre {x_var} y {y_var}")
    st.pyplot(fig)

    # Interpretación automática
    st.markdown(f"🧠 **Interpretación automática:** {interpretar_correlacion(corr)}")

    st.divider()

    # ==============================
    # 🔹 MATRIZ DE CORRELACIONES GLOBAL
    # ==============================
    st.markdown("### 🌐 Matriz global de correlaciones")

    # --- Slider para filtrar correlaciones ---
    umbral = st.slider("Umbral mínimo de correlación a mostrar (|r| ≥ ...)", 0.0, 1.0, 0.5, 0.05)

    corr_matrix = df.corr(numeric_only=True)

    # --- Enmascarar correlaciones débiles ---
    mask = corr_matrix.abs() >= umbral
    corr_filtrado = corr_matrix.where(mask)

    # --- Gráfico interactivo con Plotly ---
    fig2 = px.imshow(
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

    fig2.update_layout(
        width=900,
        height=700,
        margin=dict(l=60, r=30, t=50, b=30),
        coloraxis_colorbar=dict(title="Correlación", len=0.75),
        font=dict(size=10)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ==============================
    # 🔹 TABLA DE CORRELACIONES FUERTES
    # ==============================
    st.markdown("### 🧮 Correlaciones más significativas")

    top_corrs = (
        corr_matrix.unstack()
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
            "📊 Valores cercanos a **+1** indican relación directa fuerte (ambas aumentan juntas).  \n"
            "Valores cercanos a **-1** indican relación inversa (una sube, la otra baja)."
        )
    else:
        st.info(f"✅ No se detectaron correlaciones con |r| ≥ {umbral}.")

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

    tipo = st.selectbox("Selecciona tipo de gráfico:", ["Boxplot", "Heatmap", "Violinplot"],key="vis_tipo")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    fig, ax = plt.subplots(figsize=(8, 5))

    if tipo == "Boxplot":
        x = st.selectbox("Eje X (categoría):", cat_cols, key="vis_x")
        y = st.selectbox("Eje Y (numérico):", numeric_cols, key="vis_y")
        sns.boxplot(data=df, x=x, y=y, ax=ax)
        st.markdown("🧠 **Interpretación:** El boxplot permite identificar asimetrías y outliers en la distribución por categoría.")

    elif tipo == "Heatmap":
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.markdown("🧠 **Interpretación:** El mapa de calor muestra correlaciones fuertes o débiles entre variables numéricas.")

    elif tipo == "Violinplot":
        x = st.selectbox("Eje X (categoría):", cat_cols)
        y = st.selectbox("Eje Y (numérico):", numeric_cols)
        sns.violinplot(data=df, x=x, y=y, ax=ax)
        st.markdown("🧠 **Interpretación:** El violínplot combina boxplot y densidad, mostrando la forma completa de la distribución.")

    st.pyplot(fig)


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
        "Visualizaciones"
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
