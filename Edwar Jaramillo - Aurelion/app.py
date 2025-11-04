import streamlit as st
from modules import documentacion_view, minimarket_view, limpieza_view, estadisticas_view

st.set_page_config(page_title="Mi Minimarket - Aurelion", layout="wide")

menu = st.sidebar.radio(
    "🧭 Navegación principal",
    (
        "🏪 Mi Minimarket",
        "📘 Documentación",
        "🧹 Limpieza y transformación",
        "📊 Análisis estadístico y visualización"
    )
)

if menu == "🏪 Mi Minimarket":
    minimarket_view.mostrar_minimarket()
elif menu == "📘 Documentación":
    documentacion_view.mostrar_documentacion()
elif menu == "🧹 Limpieza y transformación":
    # Si ya hay datasets cargados en memoria
    if "datasets" in st.session_state:
        limpieza_view.mostrar_limpieza_datos(st.session_state["datasets"])
    else:
        st.warning("⚠️ No hay datasets cargados en memoria. Ve primero al módulo 'Mi Minimarket'.")
elif menu == "📊 Análisis estadístico y visualización":
    estadisticas_view.mostrar_estadisticas(st.session_state["datasets"])
