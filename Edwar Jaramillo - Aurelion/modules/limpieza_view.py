import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from modules.utils.data_loader import guardar_excel
from modules.utils.data_cleaner import eliminar_duplicados, eliminar_nulos, rellenar_nulos
from modules.utils.plot_utils import grafico_boxplot
from modules.utils.data_explorer import DataExplorer

sns.set(style="whitegrid")

# ============================================================
# 🔍 NUEVA FUNCIÓN: Exploración general
# ============================================================
def exploracion_general(datasets):
    """Explora las tablas cargadas en memoria mediante acciones seleccionables y gráficos automáticos."""
    st.subheader("📘 Exploración general de los DataFrames en memoria")
    st.write("Selecciona las tablas y acciones a ejecutar, y obtén una vista tabular y visual del contenido.")

    # -----------------------------
    # SELECCIÓN DE TABLAS Y ACCIONES
    # -----------------------------
    tablas_disponibles = list(datasets.keys())
    tablas_seleccionadas = st.multiselect(
        "Selecciona las tablas a analizar:",
        tablas_disponibles,
        default=tablas_disponibles[:1]
    )

    acciones_disponibles = {
        "Describe": "describe",
        "Info": "info",
        "Head": "head",
        "Shape": "shape",
        "Columnas": "columns"
    }

    acciones_seleccionadas = st.multiselect(
        "Selecciona las acciones a ejecutar:",
        list(acciones_disponibles.keys()),
        default=["Head"]
    )

    # ======================================================
    # 1️⃣ RESULTADOS TABULARES (acciones tipo describe/info)
    # ======================================================
    if st.button("▶ Ejecutar acciones de exploración"):
        acciones_finales = [acciones_disponibles[a] for a in acciones_seleccionadas]
        explorer = DataExplorer(datasets)
        resultados = explorer.explorar(tablas_seleccionadas, acciones_finales)

        for tabla, acciones in resultados.items():
            st.markdown(f"### 📊 Resultados para **{tabla}**")
            for accion, resultado in acciones.items():
                st.markdown(f"**🔹 Acción:** `{accion}`")
                st.dataframe(resultado, use_container_width=True)
                st.divider()

    st.markdown("---")

    # ======================================================
    # 2️⃣ VISUALIZACIÓN AUTOMÁTICA DE VARIABLES
    # ======================================================
    st.subheader("🎨 Visualización automática de datos")

    tabla_visual = st.selectbox("Selecciona una tabla para graficar:", tablas_disponibles)
    df = datasets[tabla_visual]

    tipo_grafico = st.radio(
        "Selecciona tipo de gráfico:",
        ["Histograma", "Boxplot", "Correlación (Heatmap)"],
        horizontal=True
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # --- HISTOGRAMA ---
    if tipo_grafico == "Histograma":
        col = st.selectbox("Selecciona columna numérica:", numeric_cols)
        if col:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(df[col].dropna(), kde=True, color="skyblue", ax=ax)
            ax.set_title(f"Distribución de {col}", fontsize=12)
            st.pyplot(fig)
            st.info(f"📈 El histograma muestra cómo se distribuyen los valores de **{col}**.\n"
                    "Las barras altas indican concentraciones de datos en ese rango.")

    # --- BOX PLOT ---
    elif tipo_grafico == "Boxplot":
        col_y = st.selectbox("Variable numérica (Y):", numeric_cols)
        col_x = st.selectbox("Variable categórica (X):", cat_cols)
        if col_y and col_x:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=df, x=col_x, y=col_y, palette="Set2", ax=ax)
            ax.set_title(f"Distribución de {col_y} por {col_x}")
            st.pyplot(fig)
            st.info("🧩 El boxplot permite observar la mediana, los cuartiles y posibles valores atípicos por categoría.")

    # --- HEATMAP DE CORRELACIÓN ---
    elif tipo_grafico == "Correlación (Heatmap)":
        if numeric_cols:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("Mapa de correlaciones entre variables numéricas", fontsize=12)
            st.pyplot(fig)
            st.info("🔗 Los valores cercanos a **1** indican correlación positiva, "
                    "los cercanos a **-1** indican correlación negativa, "
                    "y los cercanos a **0** indican poca o nula relación.")

    st.markdown("---")
    
        # ======================================================
    # 3️⃣ INSIGHTS AUTOMÁTICOS
    # ======================================================
    st.subheader("🧠 Insights automáticos del dataset")

    if st.button("🔍 Generar insights automáticos"):
        insights = generar_insights(df)
        if insights:
            for i, text in enumerate(insights, 1):
                st.markdown(f"{i}. {text}")
        else:
            st.info("No se generaron insights: verifica que la tabla tenga datos válidos.")


    st.info("""
    💡 **Consejos de uso:**
    - Usa *Describe* o *Head* para conocer la estructura básica de las tablas.
    - Usa *Histograma* para entender la distribución de una variable numérica.
    - Usa *Boxplot* para comparar distribuciones entre categorías.
    - Usa *Correlación (Heatmap)* para descubrir relaciones entre variables numéricas.
    """)

    return datasets

import streamlit as st
import pandas as pd
import re

# ============================================================
# 🧾 Pestaña: Revisión y recategorización de productos
# ============================================================
def revisar_categorias_productos(datasets):
    st.subheader("🧾 Revisión y recategorización de productos")
    productos = datasets.get("productos", pd.DataFrame())

    if productos.empty:
        st.warning("⚠️ No hay datos disponibles en la tabla 'productos'.")
        return datasets

    # ===============================================================
    # 📋 VISTA PREVIA EDITABLE DE PRODUCTOS CON BUSCADOR
    # ===============================================================
    st.markdown("### 📋 Vista previa de productos")

    registros_ver = st.number_input("Cantidad de registros a mostrar:", 5, 100, 10)

    # --- Controles principales ---
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Recargar productos desde memoria"):
            productos = st.session_state.datasets.get("productos", pd.DataFrame())
            st.success("✅ Datos de productos recargados desde la memoria.")
    with col2:
        st.caption("Usa este botón si aplicaste cambios desde otra pestaña (limpieza, transformaciones, etc.).")

    # --- 🔍 Buscador dinámico ---
    st.markdown("**Filtrar productos por nombre o categoría:**")
    busqueda = st.text_input("Buscar...", placeholder="Ejemplo: Jugo, Bebidas, Leche...")

    if busqueda:
        productos_filtrados = productos[
            productos["nombre_producto"].str.contains(busqueda, case=False, na=False)
            | productos["categoria"].str.contains(busqueda, case=False, na=False)
        ]
        st.info(f"🔎 Se encontraron {len(productos_filtrados)} coincidencias.")
    else:
        productos_filtrados = productos

    # --- Tabla editable ---
    st.markdown("**Haz clic sobre la columna de categoría para modificarla manualmente.**")

    editable_df = st.data_editor(
        productos_filtrados.head(registros_ver),
        num_rows="dynamic",
        use_container_width=True,
        disabled=["id_producto", "nombre_producto", "precio_unitario"],
        key="editor_productos"
    )

    # --- Guardar cambios manuales ---
    if st.button("💾 Guardar cambios manuales en categorías"):
        try:
            # Actualizar solo las filas editadas visibles en el editor
            productos.update(editable_df)
            datasets["productos"] = productos
            st.session_state.datasets = datasets
            st.success("✅ Cambios manuales aplicados correctamente en el dataset en memoria.")
        except Exception as e:
            st.error(f"❌ Error al guardar los cambios: {e}")

    st.divider()


    # ===============================================================
    # 🧮 CATEGORÍAS ÚNICAS Y CONTEO
    # ===============================================================
    st.markdown("### 🧮 Análisis de categorías")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Categorías únicas:**")
        st.write(list(productos["categoria"].unique()))
    with col2:
        st.markdown("**Conteo de productos por categoría:**")
        st.dataframe(productos["categoria"].value_counts())

    st.divider()

    # ===============================================================
    # ✨ NORMALIZACIÓN DE TEXTO
    # ===============================================================
    st.markdown("### ✨ Normalización de texto en categorías")
    normalizacion_default = {
        'Lacteos': 'Lácteos',
        'Lacteo': 'Lácteos',
        'Verdura': 'Verduras',
        'Fruta': 'Frutas',
        'Cereal': 'Cereales',
        'Otros': 'Otros Productos',
        'Alimento': 'Alimentos'
    }

    reglas_texto = st.text_area(
        "Diccionario de normalización (en formato Python dict):",
        value=str(normalizacion_default),
        height=150
    )

    if st.button("⚙️ Ejecutar normalización"):
        try:
            reglas = eval(reglas_texto)
            antes = productos["categoria"].copy()
            productos["categoria"] = productos["categoria"].replace(reglas)
            cambios = (antes != productos["categoria"]).sum()
            st.success(f"✅ Normalización ejecutada. {cambios} registros actualizados.")
            datasets["productos"] = productos
            st.session_state.datasets = datasets
            st.dataframe(productos.head(registros_ver), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error al aplicar las reglas: {e}")

    st.divider()

    # ===============================================================
    # 🧠 REGLAS DE RECATEGORIZACIÓN AUTOMÁTICA
    # ===============================================================
    st.markdown("### 🧠 Reglas automáticas de recategorización")
    reglas_default = {
        "Licor|Vodka|Ron|Vino|Whisky|Fernet":"Bebidas alcoholicas",
        "Jugo|Bebida|Agua|Refresco|Gaseosa|Té|Energética|Mate|Pepsi|Cerveza|Avena|Cola|Sprite": "Bebidas",
        "Manzana|Banano|Naranja|Pera|Uva": "Bebidas",
        "Pan|Ponqué|Bizcocho|Panela|Mermelada|Manteca": "Panadería",
        "Yogur|Leche|Queso|Mantequilla": "Lácteos",
        "Arroz|Frijol|Lenteja|Cereal|Frutos secos|Garbanzos|Granola": "Granos y Cereales",
        "Tomate|Cebolla|Papa|Lechuga|Zanahoria|Verduras": "Verduras",
        "Detergente|Jabón|Limpiador|Desinfectante|Lacandina|Shampoo|Servilletas|Cepillo|Mascarilla|Limpiavidrios|Esponjas|Desodorante": "Limpieza",
        "Galleta|Chocolate|Dulce|Confite|Alfajor|Maní|Turrón|Azúcar|Caramelo|Chupetín|Stevia|Pizza|Helado|Galletitas|Chicle Menta": "Snacks y Dulces"
    }

    st.markdown("Puedes editar las reglas directamente en formato Python dict (clave = patrón regex, valor = categoría sugerida):")
    texto_reglas = st.text_area("Reglas de categorías:", value=str(reglas_default), height=250)

    if st.button("🔍 Buscar productos potencialmente mal categorizados"):
        try:
            reglas_categoria = eval(texto_reglas)
            sugerencias = []
            for patron, categoria_sugerida in reglas_categoria.items():
                mask = productos["nombre_producto"].str.contains(patron, case=False, na=False)
                df_sugerido = productos.loc[
                    mask & (productos["categoria"] != categoria_sugerida),
                    ["id_producto", "nombre_producto", "categoria"]
                ]
                if not df_sugerido.empty:
                    df_sugerido["categoria_sugerida"] = categoria_sugerida
                    sugerencias.append(df_sugerido)

            if sugerencias:
                sugerencias_df = pd.concat(sugerencias, ignore_index=True)
                st.session_state["sugerencias_df"] = sugerencias_df
                st.success(f"🔍 Se encontraron {len(sugerencias_df)} productos potencialmente mal categorizados.")
                st.dataframe(
                    sugerencias_df,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("✅ No se detectaron productos fuera de su categoría esperada.")

        except Exception as e:
            st.error(f"❌ Error en las reglas de categorías: {e}")

    st.divider()

    # ===============================================================
    # 💾 APLICAR CAMBIOS SUGERIDOS
    # ===============================================================
    if "sugerencias_df" in st.session_state:
        sugerencias_df = st.session_state["sugerencias_df"]

        if st.button("💾 Aplicar todas las sugerencias al dataset"):
            for _, fila in sugerencias_df.iterrows():
                productos.loc[productos["id_producto"] == fila["id_producto"], "categoria"] = fila["categoria_sugerida"]
            datasets["productos"] = productos
            st.session_state.datasets = datasets
            st.success(f"✅ Cambios aplicados correctamente. {len(sugerencias_df)} registros actualizados.")
            del st.session_state["sugerencias_df"]

    # ===============================================================
    # 📈 RESUMEN FINAL
    # ===============================================================
    st.markdown("### 📈 Resumen final de categorías actualizadas")
    conteo = productos["categoria"].value_counts()
    st.dataframe(conteo)

    for cat in productos["categoria"].unique():
        subset = productos[productos["categoria"] == cat]
        st.markdown(f"#### {cat} ({len(subset)} productos)")
        st.dataframe(subset[["id_producto", "nombre_producto", "categoria"]].head(20), use_container_width=True)

    return datasets


    # ---------------------------------------------
    # APLICAR CAMBIOS
    # ---------------------------------------------
    if "sugerencias_df" in st.session_state:
        sugerencias_df = st.session_state["sugerencias_df"]

        if st.button("💾 Aplicar todas las sugerencias al dataset"):
            for _, fila in sugerencias_df.iterrows():
                productos.loc[productos["id_producto"] == fila["id_producto"], "categoria"] = fila["categoria_sugerida"]
            datasets["productos"] = productos
            st.success(f"✅ Cambios aplicados correctamente. {len(sugerencias_df)} registros actualizados.")
            del st.session_state["sugerencias_df"]

    # ---------------------------------------------
    # RESUMEN FINAL
    # ---------------------------------------------
    st.markdown("### 📈 Resumen final de categorías actualizadas")
    conteo = productos["categoria"].value_counts()
    st.dataframe(conteo)

    for cat in productos["categoria"].unique():
        subset = productos[productos["categoria"] == cat]
        st.markdown(f"#### {cat} ({len(subset)} productos)")
        st.dataframe(subset[["id_producto", "nombre_producto", "categoria"]].head(20), use_container_width=True)

    return datasets


# ============================================================
# ⚙️ FUNCIONES AUXILIARES DE LIMPIEZA (sin cambios)
# ============================================================

def mostrar_valores_faltantes(df):
    st.subheader("🔍 Detección y tratamiento de valores faltantes o vacíos")

    st.info("""
    **¿Qué hace este análisis?**  
    Esta herramienta detecta valores **nulos (`NaN`)**, **vacíos** o **con solo espacios en blanco**.  
    Puedes revisar las filas afectadas y aplicar distintas estrategias:
    - 🗑️ Eliminar filas incompletas  
    - 🧮 Rellenar con `0`, mediana o texto personalizado
    """)

    # ---------------------------------------------------------
    # 1️⃣ Reemplazar espacios vacíos por NaN para unificar criterios
    # ---------------------------------------------------------
    df = df.copy()
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # ---------------------------------------------------------
    # 2️⃣ Calcular cantidad de valores faltantes por columna
    # ---------------------------------------------------------
    nulos_total = df.isnull().sum()
    st.write("**Conteo de valores faltantes o vacíos por columna:**")
    st.dataframe(nulos_total)

    columnas_con_nulos = nulos_total[nulos_total > 0].index.tolist()

    if not columnas_con_nulos:
        st.success("✅ No se encontraron valores faltantes ni vacíos en este DataFrame.")
        return df

    # ---------------------------------------------------------
    # 3️⃣ Mostrar las filas con valores faltantes
    # ---------------------------------------------------------
    filas_afectadas = df[df[columnas_con_nulos].isnull().any(axis=1)]
    st.markdown("### 🧩 Filas con valores faltantes o vacíos")
    st.dataframe(filas_afectadas, use_container_width=True, height=300)

    # ---------------------------------------------------------
    # 4️⃣ Seleccionar estrategia de tratamiento
    # ---------------------------------------------------------
    estrategia = st.selectbox(
        "Selecciona una acción de tratamiento:",
        [
            "Ninguna",
            "Eliminar filas con valores faltantes",
            "Rellenar con 0",
            "Rellenar con mediana (numéricas)",
            "Rellenar con texto fijo"
        ]
    )

    texto_fijo = None
    if estrategia == "Rellenar con texto fijo":
        texto_fijo = st.text_input("Ingrese el texto con el que desea rellenar:")

    # ---------------------------------------------------------
    # 5️⃣ Aplicar acción seleccionada
    # ---------------------------------------------------------
    if st.button("⚙️ Aplicar acción de limpieza"):
        filas_antes = len(df)

        try:
            if estrategia == "Eliminar filas con valores faltantes":
                df = df.dropna()
                filas_despues = len(df)
                eliminadas = filas_antes - filas_despues
                st.success(f"✅ Filas con valores faltantes eliminadas ({eliminadas} filas eliminadas).")

            elif estrategia == "Rellenar con 0":
                df = df.fillna(0)
                st.success("✅ Valores faltantes reemplazados por 0.")

            elif estrategia == "Rellenar con mediana (numéricas)":
                for col in columnas_con_nulos:
                    if np.issubdtype(df[col].dtype, np.number):
                        mediana = df[col].median()
                        df[col] = df[col].fillna(mediana)
                st.success("✅ Valores numéricos faltantes reemplazados con la mediana de cada columna.")

            elif estrategia == "Rellenar con texto fijo" and texto_fijo is not None:
                df = df.fillna(texto_fijo)
                st.success(f"✅ Valores faltantes reemplazados con el texto '{texto_fijo}'.")

            else:
                st.info("ℹ️ No se aplicaron cambios al DataFrame.")

        except Exception as e:
            st.error(f"❌ Error al aplicar la acción: {e}")

        # ---------------------------------------------------------
        # 6️⃣ Vista previa después de aplicar la acción
        # ---------------------------------------------------------
        st.markdown("### 🧾 Vista previa después de la limpieza")
        st.dataframe(df.head(), use_container_width=True)

    return df




def mostrar_duplicados(df):
    st.subheader("📋 Detección y manejo de registros duplicados")

    duplicados_mask = df.duplicated(keep=False)
    duplicados_df = df[duplicados_mask]

    st.write(f"**Registros duplicados detectados:** {duplicados_df.shape[0]}")

    if duplicados_df.empty:
        st.success("✅ No hay registros duplicados.")
        return df

    st.markdown("### 🔎 Filas duplicadas detectadas")
    st.dataframe(duplicados_df, use_container_width=True, height=300)

    opcion = st.selectbox(
        "Acción a realizar:",
        ["Ninguna", "Eliminar duplicados (mantener la primera aparición)", "Eliminar todos los duplicados"]
    )

    if st.button("⚙️ Aplicar acción sobre duplicados"):
        try:
            if opcion == "Eliminar duplicados (mantener la primera aparición)":
                df = df.drop_duplicates(keep="first")
                st.success("✅ Duplicados eliminados (manteniendo la primera aparición).")
            elif opcion == "Eliminar todos los duplicados":
                df = df.drop_duplicates(keep=False)
                st.success("✅ Todos los duplicados eliminados.")
            else:
                st.info("No se aplicaron cambios.")
        except Exception as e:
            st.error(f"❌ Error al procesar duplicados: {e}")

    st.divider()
    st.markdown("### 🧾 Vista previa después del tratamiento")
    st.dataframe(df.head(), use_container_width=True)
    return df



def mostrar_inconsistencias(df):
    st.subheader("🧾 Revisión de inconsistencias y tipos de datos")
    st.write("**Tipos de datos detectados:**")
    st.dataframe(df.dtypes)

    # --- Normalización de texto ---
    columnas_texto = df.select_dtypes(include=["object"]).columns.tolist()
    if columnas_texto:
        st.markdown("### 🧹 Limpieza y normalización de texto")

        st.info("""
        **¿Qué hace la normalización de texto?**  
        La normalización estandariza el formato de las palabras en una columna de texto para mejorar la consistencia de los datos.  
        - Elimina espacios extra al inicio o final.  
        - Convierte todo el texto a formato **Título** (primera letra en mayúscula).  
        - Asegura que valores similares se escriban igual, evitando errores como “lacteos”, “Lacteos”, “Lácteos”.

        👉 Ejemplo:  
        `  leche descremada ` → `Leche Descremada`  
        ` YOGURT  ` → `Yogurt`
        """)

        col_texto = st.selectbox("Selecciona una columna de texto para normalizar:", columnas_texto)

        if st.button("🧼 Normalizar texto seleccionado"):
            antes = df[col_texto].copy()
            df[col_texto] = df[col_texto].astype(str).str.strip().str.title()
            cambios = (antes != df[col_texto]).sum()
            st.success(f"✅ Normalización aplicada correctamente. {cambios} registros modificados.")

    # --- Conversión de fecha ---
    columnas_candidatas = [col for col in df.columns if "fecha" in col.lower()]
    if columnas_candidatas:
        st.markdown("### 🗓️ Conversión de columnas de fecha")
        col_fecha = st.selectbox("Selecciona la columna a convertir:", columnas_candidatas)
        formato = st.text_input("Formato de fecha esperado (ej: %Y-%m-%d, %d/%m/%Y):", value="%Y-%m-%d")

        if st.button("🕓 Convertir a tipo datetime"):
            try:
                df[col_fecha] = pd.to_datetime(df[col_fecha], errors="coerce", format=formato)
                errores = df[col_fecha].isnull().sum()
                st.success(f"✅ Conversión realizada con éxito. {errores} registros no pudieron convertirse.")
            except Exception as e:
                st.error(f"❌ Error al convertir fechas: {e}")

    # --- Detectar valores no válidos en columnas numéricas ---
    columnas_num = df.select_dtypes(include=[np.number]).columns.tolist()
    if columnas_num:
        st.markdown("### ⚠️ Valores inconsistentes en columnas numéricas")
        st.write("Verifica si hay valores negativos o fuera de rango.")
        for col in columnas_num:
            if (df[col] < 0).any():
                st.warning(f"🚨 Columna '{col}' contiene valores negativos.")
                st.dataframe(df[df[col] < 0][[col]])

    st.divider()
    st.markdown("### 🧾 Vista previa después del tratamiento")
    st.dataframe(df.head(), use_container_width=True)
    return df



def mostrar_valores_atipicos(df):
    st.subheader("📊 Detección y manejo de valores atípicos")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No hay columnas numéricas disponibles para analizar.")
        return df

    col = st.selectbox("Selecciona una columna numérica:", numeric_cols)
    st.pyplot(grafico_boxplot(df, col))

    if st.button("Eliminar valores fuera del rango IQR"):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        st.success("✅ Valores atípicos eliminados según método IQR.")
    return df


def mostrar_transformaciones(df):
    st.subheader("🧩 Transformaciones básicas del dataset")

    col = st.selectbox("Selecciona una columna numérica para transformar:", df.select_dtypes(include=[np.number]).columns)
    if st.checkbox("Aplicar transformación logarítmica"):
        df[col] = np.log1p(df[col])
        st.success(f"Transformación logarítmica aplicada sobre '{col}'.")

    if st.checkbox("Ordenar valores"):
        orden = st.radio("Orden:", ["Ascendente", "Descendente"])
        df = df.sort_values(by=col, ascending=(orden == "Ascendente"))
        st.success(f"Datos ordenados por '{col}' en orden {orden.lower()}.")

    st.dataframe(df.head())
    return df


def generar_insights(df):
    """
    Genera insights automáticos y educativos sobre un DataFrame.
    Usa medidas estadísticas básicas y reglas simples de interpretación.
    """
    insights = []

    # --- Variables numéricas ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        for col in num_cols:
            serie = df[col].dropna()
            if serie.empty:
                continue

            mean = serie.mean()
            median = serie.median()
            std = serie.std()
            skew = serie.skew()

            # Tendencia central
            if abs(skew) < 0.3:
                insights.append(f"📊 La variable **{col}** tiene una distribución aproximadamente simétrica (media ≈ mediana).")
            elif skew > 0.3:
                insights.append(f"➡️ La variable **{col}** presenta **sesgo a la derecha**, con valores altos más dispersos.")
            else:
                insights.append(f"⬅️ La variable **{col}** presenta **sesgo a la izquierda**, con valores bajos más dispersos.")

            # Dispersión relativa
            coef_var = (std / mean) if mean != 0 else 0
            if coef_var < 0.2:
                insights.append(f"🔹 Los valores de **{col}** son bastante homogéneos (poca variabilidad).")
            elif coef_var < 0.5:
                insights.append(f"🔸 La variable **{col}** muestra una variabilidad moderada en sus datos.")
            else:
                insights.append(f"⚠️ La variable **{col}** tiene alta dispersión: los datos varían considerablemente entre registros.")

    # --- Variables categóricas ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        for col in cat_cols:
            counts = df[col].value_counts(dropna=False)
            top_val, top_freq = counts.index[0], counts.iloc[0]
            perc = (top_freq / len(df)) * 100
            if perc > 60:
                insights.append(f"🏷️ En la columna **{col}**, el valor **'{top_val}'** representa el {perc:.1f}% de los registros — alta concentración.")
            elif perc > 30:
                insights.append(f"🧩 En la columna **{col}**, el valor más frecuente es **'{top_val}'** ({perc:.1f}% de los registros).")
            else:
                insights.append(f"📦 La variable **{col}** presenta una distribución equilibrada entre categorías (sin un valor dominante).")

    # --- Correlaciones ---
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        corr_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
        )
        if not corr_pairs.empty:
            top_corr = corr_pairs.head(3)
            for (v1, v2), val in top_corr.items():
                if val > 0.7:
                    insights.append(f"🔗 Existe una correlación **fuerte** ({val:.2f}) entre **{v1}** y **{v2}**.")
                elif val > 0.4:
                    insights.append(f"📈 Se observa una correlación **moderada** ({val:.2f}) entre **{v1}** y **{v2}**.")
                else:
                    insights.append(f"⚪ Correlación débil ({val:.2f}) entre **{v1}** y **{v2}**.")

    # --- Tamaño del dataset ---
    filas, cols = df.shape
    insights.append(f"🧮 El dataset analizado tiene **{filas} filas** y **{cols} columnas**.")

    return insights

import os
import pandas as pd
from modules.utils.data_loader import guardar_excel


# ============================================================
# 💾 FUNCIÓN AUXILIAR: Guardar cambios en base de datos
# ============================================================
def guardar_cambios_base_datos(datasets):
    """Guarda todos los DataFrames en memoria dentro de database/db_limpia/."""
    st.subheader("💾 Guardar cambios en base de datos")
    st.markdown("""
    Guarda los DataFrames actualmente cargados en memoria dentro de la carpeta:
    **`database/db_limpia/`**, con el nombre `*_actualizado.xlsx`.
    """)

    # Crear carpeta si no existe
    ruta_destino = "database/db_limpia"
    os.makedirs(ruta_destino, exist_ok=True)

    # Mostrar resumen de tablas
    st.markdown("### 🧮 Resumen de tablas en memoria:")
    resumen = []
    for nombre, df in datasets.items():
        filas, cols = df.shape
        resumen.append({"Tabla": nombre, "Filas": filas, "Columnas": cols})
    st.dataframe(pd.DataFrame(resumen), use_container_width=True)

    # Botón principal
    if st.button("💾 Guardar todas las tablas en base de datos"):
        for nombre, df in datasets.items():
            ruta = os.path.join(ruta_destino, f"{nombre}_actualizado.xlsx")
            try:
                guardar_excel(df, ruta)
                st.success(f"✅ {nombre} guardada como `{nombre}_actualizado.xlsx`")
            except Exception as e:
                st.error(f"❌ Error al guardar {nombre}: {e}")

        st.info(f"📁 Archivos actualizados disponibles en `{ruta_destino}/`")

    st.markdown("---")
    st.caption("💡 Consejo: Verifica que los datos estén actualizados antes de guardar.")


# ============================================================
# 🧭 FUNCIÓN PRINCIPAL DEL MÓDULO
# ============================================================
def mostrar_limpieza_datos(datasets):
    """
    Vista principal del módulo de limpieza.
    Recibe un diccionario de DataFrames cargados en memoria:
        {
            "productos": df_productos,
            "clientes": df_clientes,
            "ventas": df_ventas,
            "detalle_ventas": df_detalle
        }
    """
    st.title("🧹 Módulo de Limpieza y Transformación de Datos")

    tabs = st.tabs([
        "Exploración\n\ngeneral",
        "Valores\n\nfaltantes",
        "Duplicados",
        "Inconsistencias",
        "Valores\n\natípicos",
        "Transformaciones",
        "Revisión de categorías\n\nde productos",
        "Guardar cambios\n\nen base de datos"
    ])

    # 1️⃣ Exploración general
    with tabs[0]:
        exploracion_general(datasets)

    # 2️⃣ Selección de tabla para limpieza
    st.sidebar.subheader("⚙️ Selección de tabla para limpieza")
    tabla_limpieza = st.sidebar.selectbox(
        "Selecciona tabla base:",
        list(datasets.keys()),
        index=3
    )
    df = datasets[tabla_limpieza]

    # -------------------------------------------------------
    # 🔹 Pestaña: Valores faltantes
    # -------------------------------------------------------
    with tabs[1]:
        df_actualizado = mostrar_valores_faltantes(df)
        if not df_actualizado.equals(df):
            datasets[tabla_limpieza] = df_actualizado
            st.session_state.datasets = datasets
            st.success(f"💾 Cambios aplicados a '{tabla_limpieza}' en memoria.")

    # -------------------------------------------------------
    # 🔹 Pestaña: Duplicados
    # -------------------------------------------------------
    with tabs[2]:
        df_actualizado = mostrar_duplicados(datasets[tabla_limpieza])
        if not df_actualizado.equals(datasets[tabla_limpieza]):
            datasets[tabla_limpieza] = df_actualizado
            st.session_state.datasets = datasets
            st.success(f"💾 Cambios aplicados a '{tabla_limpieza}' en memoria.")

    # -------------------------------------------------------
    # 🔹 Pestaña: Inconsistencias
    # -------------------------------------------------------
    with tabs[3]:
        df_actualizado = mostrar_inconsistencias(datasets[tabla_limpieza])
        if not df_actualizado.equals(datasets[tabla_limpieza]):
            datasets[tabla_limpieza] = df_actualizado
            st.session_state.datasets = datasets
            st.success(f"💾 Cambios aplicados a '{tabla_limpieza}' en memoria.")

    # -------------------------------------------------------
    # 🔹 Pestaña: Valores atípicos
    # -------------------------------------------------------
    with tabs[4]:
        df_actualizado = mostrar_valores_atipicos(datasets[tabla_limpieza])
        if not df_actualizado.equals(datasets[tabla_limpieza]):
            datasets[tabla_limpieza] = df_actualizado
            st.session_state.datasets = datasets
            st.success(f"💾 Cambios aplicados a '{tabla_limpieza}' en memoria.")

    # -------------------------------------------------------
    # 🔹 Pestaña: Transformaciones
    # -------------------------------------------------------
    with tabs[5]:
        df_actualizado = mostrar_transformaciones(datasets[tabla_limpieza])
        if not df_actualizado.equals(datasets[tabla_limpieza]):
            datasets[tabla_limpieza] = df_actualizado
            st.session_state.datasets = datasets
            st.success(f"💾 Cambios aplicados a '{tabla_limpieza}' en memoria.")

    # -------------------------------------------------------
    # 🔹 Pestaña: Revisión de categorías
    # -------------------------------------------------------
    with tabs[6]:
        datasets = revisar_categorias_productos(datasets)
        st.session_state.datasets = datasets

    # -------------------------------------------------------
    # 🔹 Pestaña: Guardar cambios
    # -------------------------------------------------------
    with tabs[7]:
        guardar_cambios_base_datos(datasets)

    # 3️⃣ Guardar versión limpia
    #if st.button("💾 Guardar versión limpia"):
    #    guardar_excel(df, f"database/db_limpia/{tabla_limpieza}_actualizado.xlsx")
    #    st.success(f"Archivo guardado como '{tabla_limpieza}_actualizado.xlsx'.")
