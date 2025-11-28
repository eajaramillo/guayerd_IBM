import streamlit as st
from modules import documentacion_view, minimarket_view, limpieza_view, estadisticas_view, reportes_view, machine_learning_view as ml_view
import modules.prediccion_ventas_view as pred_view

st.set_page_config(page_title="Mi Minimarket - Aurelion", layout="wide")

mensaje_error_dataset = "⚠️ No hay datasets cargados en memoria. Ve primero al módulo 'Mi Minimarket'."

menu = st.sidebar.radio(
    "🧭 Navegación principal",
    (
        "🏪 Mi Minimarket",
        "📘 Documentación",
        "🧹 Limpieza y transformación",
        "📈 Análisis estadístico y visualización",
        "📊 Reportes gerenciales y KPIs",
        "Machine Learning - Ejemplo",
        "ML - Predicción de Ventas"
    )
)

if menu == "🏪 Mi Minimarket":
    minimarket_view.mostrar_minimarket()
elif menu == "📘 Documentación":
    documentacion_view.mostrar_documentacion()
elif menu == "🧹 Limpieza y transformación":
    if "datasets" in st.session_state:
        limpieza_view.mostrar_limpieza_datos(st.session_state["datasets"])
    else:
        st.warning("")

elif menu == "📈 Análisis estadístico y visualización":
    if "datasets" in st.session_state:
        estadisticas_view.mostrar_estadisticas(st.session_state["datasets"])
    else:
        st.warning(mensaje_error_dataset)

elif menu == "📊 Reportes gerenciales y KPIs":
    if "datasets" in st.session_state:
        reportes_view.mostrar_reportes(st.session_state["datasets"])
    else:
        st.warning(mensaje_error_dataset)
elif menu == "Machine Learning - Ejemplo":
    ml_view.mostrar_machine_learning_view(st.session_state["datasets"])
elif menu == "ML - Predicción de Ventas":
    pred_view.mostrar_prediccion_ventas_view(st.session_state["datasets"])
