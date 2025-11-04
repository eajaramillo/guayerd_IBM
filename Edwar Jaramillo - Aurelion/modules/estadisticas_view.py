import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    tabla_seleccionada = st.selectbox(
        "Selecciona la tabla para analizar:",
        list(datasets.keys())
    )
    df = datasets[tabla_seleccionada]

    st.markdown(f"### Analizando tabla: `{tabla_seleccionada}`")

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
