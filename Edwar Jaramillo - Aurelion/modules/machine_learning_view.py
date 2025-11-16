import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from modules.utils.data_master import construir_tabla_maestra


# ============================================================
# 1️⃣ PREPARACIÓN DE DATOS
# ============================================================
def preparar_dataset_ml(datasets):
    """Construye la tabla maestra e incluye una columna 'baja_rotacion' configurable."""
    maestra = construir_tabla_maestra(datasets, mostrar_mensajes=False)
    if maestra.empty:
        st.warning("⚠️ No se pudo construir la tabla maestra.")
        return pd.DataFrame()

    # Control de margen mediante slider
    st.sidebar.markdown("### ⚙️ Configuración de datos ML")
    margen = st.sidebar.slider("Define el margen de rentabilidad (%)", 5, 80, 30)
    maestra["costo_unitario"] = maestra["precio_unitario"] * (1 - margen / 100)
    maestra["ganancia_unitaria"] = maestra["precio_unitario"] - maestra["costo_unitario"]

    # Calcular promedio de ventas por producto
    ventas_por_prod = maestra.groupby("nombre_producto")["cantidad"].sum()
    umbral_baja = st.sidebar.slider("Umbral para baja rotación (cantidad total)", 5, 200, 20)

    maestra["baja_rotacion"] = maestra["nombre_producto"].map(
        lambda p: ventas_por_prod.get(p, 0) < umbral_baja
    )

    st.success("✅ Dataset preparado correctamente para Machine Learning")
    return maestra


# ============================================================
# 2️⃣ SECCIÓN EDUCATIVA
# ============================================================
def mostrar_conceptos_educativos():
    """Explica los tipos de aprendizaje con ejemplos visuales."""
    st.markdown("## 🎓 Fundamentos de Machine Learning")
    st.write("""
    En Machine Learning, existen **tres tipos de aprendizaje principales**:
    - **Supervisado:** el modelo aprende con datos etiquetados (ej: predecir si un producto tiene baja rotación).
    - **No supervisado:** el modelo agrupa datos similares sin etiquetas (ej: segmentar clientes).
    - **Por refuerzo:** aprende de la experiencia (no se aplica aún en este proyecto).

    En Aurelion usamos aprendizaje **supervisado**, prediciendo variables de negocio
    como *baja_rotación* o *ventas esperadas* a partir de información histórica.
    """)

    fig = px.scatter(
        x=[1, 2, 3, 4, 5],
        y=[2, 3, 5, 4, 6],
        title="Ejemplo de relación supervisada (cantidad vs ventas)"
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 3️⃣ ENTRENAMIENTO DEL MODELO
# ============================================================
def entrenar_modelo_decision_tree(df):
    """Permite seleccionar variables y entrenar un árbol de decisión."""
    st.markdown("## 🌳 Entrenamiento del Árbol de Decisión")

    # Variables por defecto
    pred_default = ["cantidad", "importe_total", "mes"]
    if "baja_rotacion" not in df.columns:
        st.warning("⚠️ Falta la columna 'baja_rotacion'. Prepara los datos primero.")
        return

    # Selección de variables
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    x_cols = st.multiselect("Selecciona variables predictoras (X):", all_numeric, default=[c for c in pred_default if c in all_numeric])
    y_col = st.selectbox("Selecciona variable objetivo (y):", ["baja_rotacion"])

    if not x_cols:
        st.info("Selecciona al menos una variable predictora para continuar.")
        return

    # Preparar datos
    X = df[x_cols]
    y = df[y_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_train, y_train)

    # Predicciones y métricas
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("🎯 Precisión del modelo", f"{acc*100:.2f}%")

    # Visualización del árbol
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, feature_names=x_cols, class_names=["Alta", "Baja"], filled=True, fontsize=8)
    st.pyplot(fig)

    # Mostrar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Alta", "Baja"], yticklabels=["Alta", "Baja"], ax=ax2)
    ax2.set_xlabel("Predicción")
    ax2.set_ylabel("Real")
    st.pyplot(fig2)

    # Reporte
    st.text("📄 Reporte de clasificación:")
    st.text(classification_report(y_test, y_pred))

    return clf


# ============================================================
# 4️⃣ FUNCIÓN PRINCIPAL (Integración)
# ============================================================
def mostrar_machine_learning_view(datasets):
    """Vista principal integrada al menú de Streamlit."""
    st.title("🤖 Módulo de Machine Learning - Proyecto Aurelion")

    tabs = st.tabs(["Preparación de datos", "Conceptos", "Entrenamiento y evaluación"])

    with tabs[0]:
        df_ml = preparar_dataset_ml(datasets)
        if not df_ml.empty:
            st.dataframe(df_ml.head(), use_container_width=True)

    with tabs[1]:
        mostrar_conceptos_educativos()

    with tabs[2]:
        if "df_ml" not in locals() or df_ml.empty:
            df_ml = preparar_dataset_ml(datasets)
        if not df_ml.empty:
            entrenar_modelo_decision_tree(df_ml)
