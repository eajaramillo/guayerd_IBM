import streamlit as st
import pandas as pd
import os
from modules import limpieza_view


# ============================================================
# UTILIDADES DE CARGA Y GUARDADO
# ============================================================

@st.cache_data
def cargar_datasets():
    """Carga datasets desde /database o crea estructuras vacías si no existen."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    database_dir = os.path.join(base_dir, "../database")

    rutas = {
        "productos": os.path.join(database_dir, "productos.xlsx"),
        "clientes": os.path.join(database_dir, "clientes.xlsx"),
        "ventas": os.path.join(database_dir, "ventas.xlsx"),
        "detalle_ventas": os.path.join(database_dir, "detalle_ventas.xlsx"),
    }

    dfs = {}
    for nombre, ruta in rutas.items():
        if os.path.exists(ruta):
            dfs[nombre] = pd.read_excel(ruta)
        else:
            st.warning(f"⚠️ No se encontró {ruta}, creando estructura vacía para '{nombre}'.")
            if nombre == "productos":
                dfs[nombre] = pd.DataFrame(columns=["nombre_producto", "categoria", "precio_unitario"])
            elif nombre == "clientes":
                dfs[nombre] = pd.DataFrame(columns=["nombre_cliente", "email", "ciudad", "fecha_alta"])
            elif nombre == "ventas":
                dfs[nombre] = pd.DataFrame(columns=["id_cliente", "medio_pago", "email"])
            elif nombre == "detalle_ventas":
                dfs[nombre] = pd.DataFrame(columns=["id_venta", "id_producto", "cantidad", "precio_unitario", "importe"])
    return dfs


def guardar_datasets(dfs):
    """Guarda los datasets actualizados en database/db_limpia/"""
    for nombre, df in dfs.items():
        ruta = os.path.join("database", "db_limpia", f"{nombre}_actualizado.xlsx")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        df.to_excel(ruta, index=False)


# ============================================================
# TABLA INTERACTIVA PAGINADA
# ============================================================
def mostrar_tabla(df, titulo):
    """
    Muestra una tabla editable y permite agregar o eliminar filas
    directamente desde la interfaz de Streamlit.
    Los cambios se guardan en memoria (session_state.datasets).
    """
    st.markdown(f"### 📘 {titulo.capitalize()}")
    
    if df.empty:
        st.info(f"No hay registros actualmente en {titulo}.")
        return

    df_editable = df.reset_index(drop=True)
    st.caption(f"Total de registros: {len(df_editable)}")

    # Clave única estable para evitar errores de duplicidad
    key_suffix = hash(titulo) % (10**6)
    
    #if "fecha_alta" in df_editable.columns:
    #    df_editable["fecha_alta"] = pd.to_datetime(df_editable["fecha_alta"], errors="coerce").dt.date


    edited_df = st.data_editor(
        df_editable,
        use_container_width=True,
        num_rows="dynamic",  # permite agregar y eliminar filas
        hide_index=True,
        key=f"editor_{titulo}_{key_suffix}"
    )

    # Guardar los cambios en memoria
    if st.button(f"💾 Guardar cambios en {titulo}", key=f"guardar_{titulo}_{key_suffix}"):
        st.session_state.datasets[titulo] = edited_df.copy()
        st.success(f"✅ Cambios actualizados en memoria para '{titulo}'.")


# ============================================================
# VISTA PRINCIPAL
# ============================================================

def mostrar_minimarket():
    st.title("🛍️ Mi Minimarket - Gestión de datos")
    st.markdown("Administra los datos de tu marketplace: carga, registra y visualiza la información.")

    # --- Cargar datasets en sesión ---
    if "datasets" not in st.session_state:
        st.session_state.datasets = cargar_datasets()

    dfs = st.session_state.datasets

    tabs_menu = st.tabs(["📝 Registro de datos", "📂 Cargar dataset", "🧹 Limpieza de datos"])

    # ============================================================
    # 📝 Pestaña 1 - REGISTRO DE DATOS
    # ============================================================
    with tabs_menu[0]:
        st.header("🧾 Registro de nuevos datos")

        tipo = st.radio(
            "Selecciona el tipo de registro a agregar:",
            ["Producto", "Cliente", "Venta", "Detalle de venta"],
            horizontal=True
        )

        # ------------------------
        # Registro de producto
        # ------------------------
        if tipo == "Producto":
            nombre = st.text_input("Nombre del producto:")
            categoria = st.text_input("Categoría:")
            precio = st.number_input("Precio unitario:", min_value=0.0, step=0.01)

            if st.button("💾 Guardar producto"):
                nuevo = pd.DataFrame(
                    [{"nombre_producto": nombre, "categoria": categoria, "precio_unitario": precio}]
                )
                dfs["productos"] = pd.concat([dfs["productos"], nuevo], ignore_index=True)
                st.session_state.datasets = dfs
                st.success(f"✅ Producto '{nombre}' agregado correctamente.")

        # ------------------------
        # Registro de cliente
        # ------------------------
        elif tipo == "Cliente":
            nombre = st.text_input("Nombre del cliente:")
            email = st.text_input("Correo electrónico:")
            ciudad = st.text_input("Ciudad:")
            #fecha_alta = st.date_input("Fecha de alta del cliente:")


            if st.button("💾 Guardar cliente"):
                nuevo = pd.DataFrame(
                    [{"nombre_cliente": nombre, "email": email, "ciudad": ciudad}]
                )
                dfs["clientes"] = pd.concat([dfs["clientes"], nuevo], ignore_index=True)
                st.session_state.datasets = dfs
                st.success(f"✅ Cliente '{nombre}' agregado correctamente.")

        # ------------------------
        # Registro de venta
        # ------------------------
        elif tipo == "Venta":
            id_cliente = st.number_input("ID Cliente:", min_value=1, step=1)
            medio_pago = st.text_input("Medio de pago:")
            email = st.text_input("Correo electrónico del cliente:")
            if st.button("💾 Guardar venta"):
                nuevo = pd.DataFrame(
                    [{"id_cliente": id_cliente, "medio_pago": medio_pago, "email": email}]
                )
                dfs["ventas"] = pd.concat([dfs["ventas"], nuevo], ignore_index=True)
                st.session_state.datasets = dfs
                st.success("✅ Venta registrada correctamente.")

        # ------------------------
        # Registro de detalle de venta
        # ------------------------
        elif tipo == "Detalle de venta":
            id_venta = st.number_input("ID Venta:", min_value=1, step=1)
            id_producto = st.number_input("ID Producto:", min_value=1, step=1)
            cantidad = st.number_input("Cantidad:", min_value=1, step=1)
            precio_unitario = st.number_input("Precio unitario:", min_value=0.0, step=0.01)
            importe = cantidad * precio_unitario

            st.write(f"💰 Importe total calculado: **{importe:,.2f}**")

            if st.button("💾 Guardar detalle de venta"):
                nuevo = pd.DataFrame(
                    [{
                        "id_venta": id_venta,
                        "id_producto": id_producto,
                        "cantidad": cantidad,
                        "precio_unitario": precio_unitario,
                        "importe": importe
                    }]
                )
                dfs["detalle_ventas"] = pd.concat([dfs["detalle_ventas"], nuevo], ignore_index=True)
                st.session_state.datasets = dfs
                st.success("✅ Detalle de venta guardado correctamente.")

    # ============================================================
    # 📂 Pestaña 2 - CARGAR DATASETS
    # ============================================================
    with tabs_menu[1]:
        st.header("📂 Cargar y visualizar datasets")

        opcion = st.radio(
            "Selecciona qué versión cargar:",
            ["Dataset original", "Última versión limpia (db_limpia)"],
            horizontal=True
        )

        if st.button("🔄 Cargar versión seleccionada"):
            if opcion == "Dataset original":
                st.session_state.datasets = cargar_datasets()
                st.success("✅ Dataset original recargado correctamente.")
            else:
                nuevas = {}
                ruta_base = "database/db_limpia"
                if os.path.exists(ruta_base):
                    for archivo in os.listdir(ruta_base):
                        if archivo.endswith(".xlsx"):
                            nombre = archivo.replace("_actualizado.xlsx", "")
                            nuevas[nombre] = pd.read_excel(os.path.join(ruta_base, archivo))
                    st.session_state.datasets = nuevas
                    st.success("✅ Última versión limpia cargada correctamente.")
                else:
                    st.warning("⚠️ No existe la carpeta database/db_limpia todavía.")

        st.markdown("---")

        st.markdown("### 🧩 Edición interactiva de datasets")

        for nombre, df in st.session_state.datasets.items():
            with st.expander(f"📘 {nombre.capitalize()} ({len(df)} registros)", expanded=False):
                mostrar_tabla(df, nombre)

        if st.button("💾 Guardar datasets en db_limpia/"):
            guardar_datasets(st.session_state.datasets)
            st.success("✅ Todos los datasets guardados correctamente en `database/db_limpia/`.")


    # ============================================================
    # 🧹 Pestaña 3 - Módulo de Limpieza
    # ============================================================
    with tabs_menu[2]:
        st.header("🧹 Limpieza y transformación de datos")
        st.markdown("Analiza, explora y limpia los datos cargados actualmente en memoria.")
        limpieza_view.mostrar_limpieza_datos(st.session_state.datasets)
