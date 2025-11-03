import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from modules.utils.data_loader import cargar_excel
from modules.utils.plot_utils import grafico_heatmap, grafico_boxplot, grafico_histograma

sns.set(style="whitegrid")


# ============================================================
# Funciones auxiliares de análisis
# ============================================================

def mostrar_estadistica_descriptiva(df):
    st.subheader("📊 Estadística descriptiva general")
    st.dataframe(df.describe())

    col = st.selectbox("Selecciona columna numérica:", df.select_dtypes(include=[np.number]).columns)
    st.write(f"**Media:** {df[col].mean():.2f}")
    st.write(f"**Mediana:** {df[col].median():.2f}")
    st.write(f"**Moda:** {df[col].mode().iloc[0]:.2f}")
    st.write(f"**Desviación estándar:** {df[col].std():.2f}")
    st.pyplot(grafico_histograma(df, col))

    st.info("""
    🔍 **Interpretación**  
    - **Media:** promedio general  
    - **Mediana:** punto medio  
    - **Moda:** valor más frecuente  
    - **Desviación estándar:** mide dispersión
    """)


def mostrar_medidas_posicion(df):
    st.subheader("📐 Medidas de posición")
    col = st.selectbox("Selecciona columna:", df.select_dtypes(include=[np.number]).columns)

    minimo = df[col].min()
    maximo = df[col].max()
    q1, q2, q3 = df[col].quantile([0.25, 0.5, 0.75])
    rango = maximo - minimo

    st.write(f"**Mínimo:** {minimo:.2f}")
    st.write(f"**Máximo:** {maximo:.2f}")
    st.write(f"**Cuartiles:** Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}")
    st.write(f"**Rango:** {rango:.2f}")
    st.pyplot(grafico_boxplot(df, col))


def mostrar_correlaciones(df):
    st.subheader("🔗 Correlaciones entre variables")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    col1 = st.selectbox("Variable 1:", numeric_cols)
    col2 = st.selectbox("Variable 2:", numeric_cols)

    corr = df[[col1, col2]].corr().iloc[0, 1]
    st.write(f"**Coeficiente de correlación (r):** {corr:.3f}")

    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=col1, y=col2, ax=ax)
    ax.set_title(f"Dispersión entre {col1} y {col2}")
    st.pyplot(fig)

    st.write("**Matriz general de correlaciones:**")
    st.pyplot(grafico_heatmap(df))


def mostrar_confiabilidad(df):
    st.subheader("🧭 Evaluación de confiabilidad")
    col = st.selectbox("Selecciona una columna numérica:", df.select_dtypes(include=[np.number]).columns)
    std = df[col].std()
    mean = df[col].mean()
    cv = (std / mean) * 100

    st.write(f"**Desviación estándar:** {std:.2f}")
    st.write(f"**Coeficiente de variación (CV):** {cv:.2f}%")

    if cv < 30:
        st.success("✅ Datos consistentes: baja dispersión")
    else:
        st.warning("⚠️ Datos con alta variabilidad")

    st.pyplot(grafico_histograma(df, col))


def mostrar_visualizaciones(df):
    st.subheader("📉 Visualizaciones estadísticas interactivas")

    tipo = st.selectbox("Selecciona tipo de gráfico:", ["Boxplot", "Heatmap", "Violinplot"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    fig, ax = plt.subplots(figsize=(8, 5))

    if tipo == "Boxplot":
        x = st.selectbox("Eje X (categoría):", cat_cols)
        y = st.selectbox("Eje Y (numérico):", numeric_cols)
        sns.boxplot(data=df, x=x, y=y, ax=ax)

    elif tipo == "Heatmap":
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)

    elif tipo == "Violinplot":
        x = st.selectbox("Eje X (categoría):", cat_cols)
        y = st.selectbox("Eje Y (numérico):", numeric_cols)
        sns.violinplot(data=df, x=x, y=y, ax=ax)

    st.pyplot(fig)

    st.info("""
    📘 **Interpretación:**  
    - *Boxplot:* muestra distribución y outliers.  
    - *Violinplot:* visualiza densidad de valores.  
    - *Heatmap:* revela correlaciones entre variables.
    """)


# ============================================================
# FUNCIÓN PRINCIPAL DEL MÓDULO
# ============================================================

def mostrar_estadisticas():
    """Vista principal del módulo de análisis y visualización."""
    st.title("📈 Análisis Estadístico y Visualización de Datos")

    df = cargar_excel("database/detalle_ventas_limpio.xlsx")

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
