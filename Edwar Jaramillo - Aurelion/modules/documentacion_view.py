# modules/documentacion_view.py
import streamlit as st
import re

# ============================================================
# Funciones auxiliares
# ============================================================

def extraer_titulos_md(contenido):
    """Extrae títulos y subtítulos del markdown."""
    titulos = []
    lineas = contenido.splitlines()
    for i, linea in enumerate(lineas):
        if linea.startswith("#"):
            nivel = len(linea.split(" ")[0])  # cantidad de #
            texto = linea.replace("#", "").strip()
            titulos.append((nivel, texto, i))
    return titulos


def extraer_seccion(contenido, inicio_linea, siguiente_linea=None):
    """Devuelve el texto entre dos títulos."""
    lineas = contenido.splitlines()
    if siguiente_linea:
        return "\n".join(lineas[inicio_linea:siguiente_linea])
    else:
        return "\n".join(lineas[inicio_linea:])


def resaltar_busqueda(texto, palabra):
    """Resalta la palabra buscada en el texto."""
    patron = re.compile(re.escape(palabra), re.IGNORECASE)
    return patron.sub(lambda m: f"<mark style='background-color:#ffd54f'>{m.group(0)}</mark>", texto)


# ============================================================
# Vista principal
# ============================================================

def mostrar_documentacion():
    st.title("📘 Documentación del Proyecto Aurelion")
    st.write("Explora, busca y navega dinámicamente el archivo `documentacion.md` en esta vista interactiva.")

    # Cargar archivo markdown
    try:
        with open("documentacion.md", "r", encoding="utf-8") as f:
            contenido = f.read()
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo 'documentacion.md'.")
        return

    # Extraer títulos
    titulos = extraer_titulos_md(contenido)
    if not titulos:
        st.warning("No se detectaron títulos en el documento.")
        st.markdown(contenido)
        return

    # Diseño en columnas: panel lateral interno + contenido
    col1, col2 = st.columns([0.35, 0.65])

    with col1:
        st.subheader("🔍 Buscar en documentación")
        busqueda = st.text_input("Palabra clave:")

        mostrar_todo = st.checkbox("📄 Ver documento completo", value=False)

        st.subheader("📑 Índice de secciones:")
        menu_titulos = []
        for nivel, texto, _ in titulos:
            indent = " " * (nivel - 1)
            menu_titulos.append(f"{indent}• {texto}")

        seleccion = st.radio("", menu_titulos, label_visibility="collapsed")

    with col2:
        # Mostrar documento completo
        if mostrar_todo:
            st.markdown("### 📄 Documento completo")
            st.markdown(contenido)
            return

        # Mostrar sección seleccionada
        titulo_limpio = seleccion.replace("•", "").strip()
        for i, (_, texto, inicio) in enumerate(titulos):
            if texto == titulo_limpio:
                siguiente = titulos[i + 1][2] if i + 1 < len(titulos) else None
                seccion = extraer_seccion(contenido, inicio, siguiente)
                if busqueda:
                    seccion = resaltar_busqueda(seccion, busqueda)
                    st.markdown(f"### 🔎 Resultados filtrados para: `{busqueda}`")
                    st.markdown(seccion, unsafe_allow_html=True)
                else:
                    st.markdown(seccion)
                break

        # Mostrar coincidencias globales si hay búsqueda
        if busqueda:
            st.markdown("---")
            st.subheader(f"📖 Coincidencias en todo el documento")
            coincidencias = [
                (i + 1, line)
                for i, line in enumerate(contenido.splitlines())
                if busqueda.lower() in line.lower()
            ]
            if coincidencias:
                for num, line in coincidencias:
                    resaltado = resaltar_busqueda(line, busqueda)
                    st.markdown(f"<small>Línea {num}</small> — {resaltado}", unsafe_allow_html=True)
            else:
                st.info("No se encontraron coincidencias.")
