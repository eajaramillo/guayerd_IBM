import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import io

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from modules.utils.data_master import construir_tabla_maestra
import plotly.graph_objects as go

if "dataset_ampliado" not in st.session_state:
    st.session_state["dataset_ampliado"] = None


def mostrar_ampliacion_dataset(datasets):
    """Vista interactiva mejorada para ampliar el dataset con mayor variabilidad, nuevos clientes y ventas más realistas."""
    st.subheader("🧩 Ampliación Avanzada del Dataset con Variabilidad Realista")

    # 1️⃣ Construir la tabla maestra base
    df = construir_tabla_maestra(datasets, mostrar_mensajes=False)
    if df.empty:
        st.warning("⚠️ No hay datos disponibles para ampliar.")
        return

    if "id_venta" not in df.columns or "cantidad" not in df.columns:
        st.error("❌ La tabla maestra debe tener las columnas 'id_venta' y 'cantidad'.")
        return

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    base_ventas = df["id_venta"].nunique()
    base_productos = df.get("nombre_producto", pd.Series()).nunique()
    base_categorias = df.get("categoria", pd.Series()).nunique()
    base_cantidades = df["cantidad"].sum()

    # 2️⃣ Factor de ampliación
    factor = st.slider("Multiplicar dataset por:", 1, 20, 1)
    st.markdown("Aumenta el tamaño del dataset simulando nuevas ventas con variaciones más amplias y realistas.")

    # 3️⃣ Generar dataset ampliado con variabilidad
    df_extendido = []
    for i in range(factor):
        df_copy = df.copy()

        # Regenerar IDs únicos
        df_copy["id_venta"] = df_copy["id_venta"].astype(str) + f"_{i+1}"

        # Desplazar fechas (simulación estacional)
        if "fecha" in df_copy.columns:
            df_copy["fecha"] = df_copy["fecha"] + timedelta(days=np.random.randint(10, 60) * i)

        # Variar importes con ruido normal (±35%)
        if "importe_total" in df_copy.columns:
            variacion_importe = np.random.normal(1.0, 0.35, len(df_copy))
            df_copy["importe_total"] = (df_copy["importe_total"] * variacion_importe).clip(lower=50).round(2)

        # 🔹 NUEVO: generar variabilidad avanzada en cantidad
        # Se basa en importe_total + ruido aleatorio + categoría
        base_cantidad = df_copy["cantidad"] * np.random.uniform(0.5, 2.0, len(df_copy))
        ruido = np.random.normal(1.0, 0.4, len(df_copy))  # más dispersión
        df_copy["cantidad"] = (base_cantidad * ruido).round().astype(int)
        df_copy["cantidad"] = df_copy["cantidad"].clip(lower=1, upper=50)  # ampliar rango máximo

        # Simular nuevos clientes
        if "cliente" in df_copy.columns:
            nuevos_clientes = [f"Cliente_{np.random.randint(1000, 9999)}" for _ in range(len(df_copy))]
            df_copy["cliente"] = np.where(np.random.rand(len(df_copy)) < 0.4, nuevos_clientes, df_copy["cliente"])

        # Variar categorías y productos
        if "categoria" in df_copy.columns:
            df_copy["categoria"] = df_copy["categoria"].apply(
                lambda c: c if np.random.rand() > 0.2 else f"{c}_Alt{i+1}"
            )

        if "nombre_producto" in df_copy.columns:
            df_copy["nombre_producto"] = df_copy["nombre_producto"].apply(
                lambda p: p if np.random.rand() > 0.1 else f"{p}_V{i+1}"
            )

        df_extendido.append(df_copy)

    df_extendido = pd.concat(df_extendido, ignore_index=True)

    # 🔹 Extra: generar comportamiento no lineal (precio bajo → más cantidad)
    if "importe_total" in df_extendido.columns:
        correlador = np.random.uniform(0.5, 1.5, len(df_extendido))
        df_extendido["cantidad"] = (
            (df_extendido["cantidad"] * (1 / np.log1p(df_extendido["importe_total"])) * 20 * correlador)
            .round().astype(int)
        )
        df_extendido["cantidad"] = df_extendido["cantidad"].clip(lower=1, upper=60)

    # 4️⃣ Calcular métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧾 Ventas únicas", f"{df_extendido['id_venta'].nunique():,}", f"x{factor}")
    col2.metric("📦 Productos", f"{df_extendido['nombre_producto'].nunique():,}")
    col3.metric("🏷️ Categorías", f"{df_extendido['categoria'].nunique():,}")
    col4.metric("📊 Total de cantidades", f"{df_extendido['cantidad'].sum():,}")

    # 5️⃣ Visualización de dispersión
    fig = px.scatter(
        df_extendido.sample(min(1000, len(df_extendido))),
        x="importe_total",
        y="cantidad",
        color="categoria" if "categoria" in df_extendido.columns else None,
        title="📉 Dispersión de Importe Total vs Cantidad Vendida",
        labels={"importe_total": "Importe Total ($)", "cantidad": "Cantidad Vendida (unidades)"},
        opacity=0.6
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # 6️⃣ Guardar en memoria
    st.session_state["dataset_ampliado"] = df_extendido
    st.success(f"✅ Dataset ampliado ({len(df_extendido)} registros) guardado en memoria global.")
    st.info("💡 Ahora las cantidades tienen mayor dispersión y relación no lineal con los importes, ideal para entrenamiento ML.")

    # 7️⃣ Vista previa
    st.markdown("### 📄 Vista previa del dataset ampliado")
    st.dataframe(df_extendido.sample(10), use_container_width=True)

    # 8️⃣ Exportación
    st.markdown("### 💾 Exportar dataset ampliado")
    buffer_csv = io.BytesIO()
    buffer_excel = io.BytesIO()
    df_extendido.to_csv(buffer_csv, index=False)
    df_extendido.to_excel(buffer_excel, index=False, sheet_name="dataset_ampliado")

    col_a, col_b = st.columns(2)
    col_a.download_button(
        label="📥 Descargar como CSV",
        data=buffer_csv.getvalue(),
        file_name="dataset_ampliado.csv",
        mime="text/csv"
    )
    col_b.download_button(
        label="📘 Descargar como Excel",
        data=buffer_excel.getvalue(),
        file_name="dataset_ampliado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if st.button("🧹 Limpiar dataset ampliado de memoria"):
        st.session_state["dataset_ampliado"] = None
        st.info("✅ Dataset ampliado eliminado de memoria.")

    return df_extendido




# ============================================================
# 1️⃣ PREPARACIÓN DE DATOS
# ============================================================

def preparar_datos_regresion(datasets):
    """Construye la tabla maestra de Aurelion y genera datos base para ML."""
    df = construir_tabla_maestra(datasets, mostrar_mensajes=False)
    if df.empty:
        st.warning("⚠️ No hay datos para análisis predictivo.")
        return pd.DataFrame()

    df["año"] = pd.to_datetime(df["fecha"], errors="coerce").dt.year
    df["mes"] = pd.to_datetime(df["fecha"], errors="coerce").dt.month

    df = df[["nombre_producto", "categoria", "cantidad", "importe_total", "mes", "año"]]
    return df


# ============================================================
# 2️⃣ GENERAR DATOS SINTÉTICOS (AMPLIAR DATASET)
# ============================================================

def ampliar_dataset(df):
    """Permite aumentar el tamaño del dataset para entrenamiento."""
    st.sidebar.subheader("🧩 Ampliar dataset de entrenamiento")
    factor = st.sidebar.slider("Multiplicar dataset por:", 1, 10, 1)
    df_extended = pd.concat([df] * factor, ignore_index=True)
    st.info(f"✅ Dataset ampliado a {len(df_extended)} registros (x{factor})")
    return df_extended


# ============================================================
# 3️⃣ REGRESIÓN: LINEAL vs KNN vs RFR
# ============================================================

def comparar_regresiones(df):
    """Compara modelos de regresión (Linear, KNN y RandomForest) e interpreta resultados."""
    st.subheader("📈 Comparación de Modelos de Regresión")
    st.markdown("Modelos incluidos: **Linear Regression**, **KNeighbors Regressor** y **Random Forest Regressor**.")
    st.write("---")

    # ============================================================
    # 1️⃣ Preparación de los datos
    # ============================================================
    X = df[["importe_total", "mes"]]
    y = df["cantidad"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ============================================================
    # 2️⃣ Entrenamiento de modelos
    # ============================================================
    lr = LinearRegression()
    knn = KNeighborsRegressor(n_neighbors=5)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    lr.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    # ============================================================
    # 3️⃣ Métricas comparativas
    # ============================================================
    resultados = pd.DataFrame({
        "Modelo": ["Linear Regression", "KNN Regressor", "Random Forest"],
        "MSE": [
            mean_squared_error(y_test, y_pred_lr),
            mean_squared_error(y_test, y_pred_knn),
            mean_squared_error(y_test, y_pred_rf)
        ],
        "R²": [
            r2_score(y_test, y_pred_lr),
            r2_score(y_test, y_pred_knn),
            r2_score(y_test, y_pred_rf)
        ]
    }).sort_values(by="R²", ascending=False)

    mejor_modelo = resultados.iloc[0]
    st.markdown("### 📋 Métricas de Evaluación")
    st.dataframe(resultados.style.format({"MSE": "{:.2f}", "R²": "{:.3f}"}))

    # --- 🔍 Explicación de las métricas ---
    with st.expander("ℹ️ ¿Qué significan estas métricas?"):
        st.markdown("""
        - **MSE (Error Cuadrático Medio)**: mide cuánto se desvía la predicción del valor real.  
          🔹 Cuanto **menor** sea el MSE, **mejor precisión** del modelo.  
          🔹 Si el MSE es alto, el modelo tiene mayor error en sus estimaciones.
        
        - **R² (Coeficiente de Determinación)**: mide qué tan bien el modelo explica la variabilidad de los datos.  
          🔹 Valores cercanos a **1.0** indican una predicción muy precisa.  
          🔹 Valores cercanos a **0.0** indican que el modelo no logra explicar bien las variaciones.
        """)

    st.success(f"🏆 Mejor modelo: **{mejor_modelo['Modelo']}** con R² = {mejor_modelo['R²']:.3f}")

    # ============================================================
    # 4️⃣ Gráfico comparativo de curvas de predicción
    # ============================================================
    X_grid = np.linspace(X["importe_total"].min(), X["importe_total"].max(), 200).reshape(-1, 1)
    X_grid_full = np.hstack((X_grid, np.full_like(X_grid, X["mes"].mean())))

    y_grid_lr = lr.predict(X_grid_full)
    y_grid_knn = knn.predict(X_grid_full)
    y_grid_rf = rf.predict(X_grid_full)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=X_train["importe_total"], y=y_train, mode="markers",
                             name="Train", opacity=0.5, marker=dict(color="gray")))
    fig.add_trace(go.Scatter(x=X_test["importe_total"], y=y_test, mode="markers",
                             name="Test", marker=dict(color="black", symbol="x")))

    fig.add_trace(go.Scatter(x=X_grid.flatten(), y=y_grid_lr, mode="lines",
                             name="Linear Regression", line=dict(width=3, color="#1f77b4")))
    fig.add_trace(go.Scatter(x=X_grid.flatten(), y=y_grid_knn, mode="lines",
                             name="KNN Regressor", line=dict(width=3, color="#2ca02c")))
    fig.add_trace(go.Scatter(x=X_grid.flatten(), y=y_grid_rf, mode="lines",
                             name="Random Forest", line=dict(width=3, color="#d62728")))

    fig.update_layout(
        title="📉 Comparación de Modelos de Regresión",
        xaxis_title="Importe Total ($)",
        yaxis_title="Cantidad Vendida (unidades)",
        legend_title="Modelos",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Texto explicativo del gráfico ---
    st.markdown("""
    **🧠 Interpretación:**
    - Cada línea representa cómo el modelo predice la cantidad vendida en función del importe total.
    - Las líneas **verdes y rojas** (KNN y Random Forest) capturan mejor las variaciones no lineales, mientras que la **azul** (Linear Regression) asume una relación más rígida.
    - Las líneas horizontales o escalonadas pueden deberse a una cantidad limitada de datos o valores repetidos en los conjuntos de prueba.
    
    💡 *Recomendación:* ampliar el dataset de entrenamiento con más variabilidad en las ventas o aplicar una normalización previa ayudará a suavizar las predicciones.
    """)

    # ============================================================
    # 5️⃣ Gráfico resumen de rendimiento
    # ============================================================
    fig_bar = px.bar(
        resultados,
        x="Modelo",
        y="R²",
        color="Modelo",
        text=resultados["R²"].apply(lambda x: f"{x:.3f}"),
        title="🔍 Comparación de rendimiento (R² por modelo)",
        color_discrete_sequence=["#1f77b4", "#2ca02c", "#d62728"]
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(yaxis_title="R² Score", xaxis_title=None)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Conclusión gerencial ---
    st.markdown("""
    ### 🏁 Conclusión Gerencial
    - **Random Forest** muestra el mejor desempeño (mayor R²), lo que significa que logra explicar mejor las variaciones en las ventas.  
    - **Linear Regression** tiene una precisión moderada, útil para entender relaciones simples entre variables.  
    - **KNN Regressor** captura patrones locales, pero puede verse afectado por la dispersión o poca densidad de datos.
    
    🔸 **Para Aurelion**, esto significa:
    - El modelo Random Forest es el más confiable para predecir las cantidades vendidas por producto o categoría.
    - Puede emplearse para **planificación de inventario, ajustes de precios y proyección de demanda.**
    - Los resultados de R² indican que todavía hay espacio para mejora (ideal > 0.8), lo cual se puede lograr **aumentando el dataset** o **incorporando nuevas variables predictoras** (como ciudad, cliente o medio de pago).
    """)

    # ============================================================
    # 6️⃣ Predicción de ventas futuras (simulación interactiva)
    # ============================================================
    st.write("---")
    st.markdown("### 🔮 Predicción de Ventas Futuras (Simulación)")
    st.markdown("""
    Usa el modelo de regresión entrenado para **predecir cuántas unidades se venderán** según un importe total estimado y un mes.  
    Puedes elegir el modelo que prefieras para comparar cómo varían los resultados.
    """)

    # --- Selección del modelo para predecir ---
    modelo_seleccionado = st.selectbox(
        "📘 Selecciona el modelo a utilizar para la predicción:",
        ["Linear Regression", "KNeighbors Regressor", "Random Forest Regressor"],
        index=2  # por defecto Random Forest
    )

    # Entradas del usuario
    col1, col2 = st.columns(2)
    importe_usuario = col1.number_input(
        "💵 Importe total estimado ($)", min_value=100.0, max_value=30000.0, value=5000.0, step=100.0
    )
    mes_usuario = col2.slider("📆 Mes proyectado", 1, 12, 6)

    # Seleccionar modelo
    if modelo_seleccionado == "Linear Regression":
        modelo = lr
    elif modelo_seleccionado == "KNeighbors Regressor":
        modelo = knn
    else:
        modelo = rf

    # Realizar predicción
    X_nueva = pd.DataFrame([[importe_usuario, mes_usuario]], columns=["importe_total", "mes"])
    prediccion = modelo.predict(X_nueva)[0]

    # Mostrar resultado
    st.metric("📦 Cantidad estimada a vender (unidades)", f"{prediccion:.2f}", help="Predicción generada por el modelo seleccionado.")

    # ============================================================
    # Gráfico interactivo de simulación
    # ============================================================
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=X["importe_total"], y=y,
        mode="markers", name="Datos históricos", opacity=0.5, marker=dict(color="gray")
    ))
    fig_pred.add_trace(go.Scatter(
        x=[importe_usuario], y=[prediccion],
        mode="markers+text", name="Predicción futura",
        text=[f"{prediccion:.2f} unidades"],
        textposition="top center",
        marker=dict(size=12, color="red", symbol="star")
    ))

    fig_pred.update_layout(
        title=f"📊 Predicción simulada con {modelo_seleccionado}",
        xaxis_title="Importe Total ($)",
        yaxis_title="Cantidad Vendida (unidades)",
        template="plotly_white"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ============================================================
    # Interpretación contextual según modelo
    # ============================================================
    if modelo_seleccionado == "Linear Regression":
        st.markdown(f"""
        **🧠 Interpretación con Linear Regression:**
        - El modelo asume una **relación lineal** entre el importe total y las unidades vendidas.
        - Tiende a generalizar bien cuando las ventas crecen de forma constante con el importe.
        - Para un importe de **${importe_usuario:,.0f}** en el mes **{mes_usuario}**, predice unas **{prediccion:.1f} unidades**.
        - Puede no capturar comportamientos atípicos o estacionales, pero es útil para una **visión global y tendencia general**.
        """)
    elif modelo_seleccionado == "KNeighbors Regressor":
        st.markdown(f"""
        **🧠 Interpretación con KNN Regressor:**
        - Este modelo se basa en **vecinos más cercanos** para estimar la cantidad vendida.
        - Captura patrones locales, pero su precisión depende de la densidad y variedad de los datos.
        - En este escenario (importe ${importe_usuario:,.0f}, mes {mes_usuario}), estima **{prediccion:.1f} unidades**.
        - Puede verse afectado si los datos históricos están agrupados o hay pocos puntos de referencia.
        """)
    else:
        st.markdown(f"""
        **🧠 Interpretación con Random Forest Regressor:**
        - Modelo basado en **múltiples árboles de decisión** que combinan resultados para mejorar la predicción.
        - Es el más robusto ante fluctuaciones y no linealidades en los datos.
        - Predice que para un importe de **${importe_usuario:,.0f}** en el mes **{mes_usuario}**, se venderán aproximadamente **{prediccion:.1f} unidades**.
        - Ideal para escenarios reales con variaciones de demanda, promociones o estacionalidad.
        """)

    # Pie explicativo
    st.markdown("""
    ---
    🧾 **Nota de interpretación:**  
    - El eje **Y (Cantidad Vendida)** está expresado en **unidades de producto vendidas**.  
    - Si las predicciones parecen discretas o “escalonadas”, se debe a la escala del dataset original.  
    - Puedes mejorar la precisión aumentando el dataset o incorporando nuevas variables predictoras (por ejemplo, ciudad o medio de pago).
    """)


    return lr, knn, rf, X_test, y_test, y_pred_lr, y_pred_knn, y_pred_rf




# ============================================================
# 4️⃣ CLASIFICACIÓN Y MATRIZ DE CONFUSIÓN
# ============================================================

def mostrar_matriz_confusion(df):
    """Entrena un modelo de clasificación (baja rotación) y muestra matriz de confusión con interpretación gerencial."""
    st.subheader("🔍 Análisis de Clasificación: Matriz de Confusión e Interpretación Gerencial")

    # ================================
    # 1️⃣ Preparación de los datos
    # ================================
    if df.empty or "cantidad" not in df.columns or "importe_total" not in df.columns:
        st.warning("⚠️ No hay suficientes datos o columnas ('cantidad', 'importe_total') para realizar la clasificación.")
        return

    # Crear columna binaria (0 = rotación normal, 1 = baja rotación)
    threshold = df["cantidad"].median()
    df["baja_rotacion"] = (df["cantidad"] < threshold).astype(int)

    X = df[["importe_total", "mes"]] if "mes" in df.columns else df[["importe_total"]]
    y = df["baja_rotacion"]

    # División del dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ================================
    # 2️⃣ Entrenamiento del modelo
    # ================================
    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # ================================
    # 3️⃣ Métricas principales
    # ================================
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.metric("🎯 Precisión del modelo", f"{acc*100:.2f}%")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="viridis", ax=ax, colorbar=True)
    plt.title("Matriz de Confusión - Baja Rotación", fontsize=13)
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Verdadera")
    st.pyplot(fig)

    # ================================
    # 4️⃣ Interpretación de resultados
    # ================================
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    total = tn + fp + fn + tp
    precision = (tp + tn) / total if total > 0 else 0

    st.markdown("### 🧠 Interpretación del Modelo de Clasificación")
    st.markdown(f"""
    El modelo predice si un producto tiene **baja rotación (1)** o **rotación normal (0)**.

    - **Verdaderos positivos (TP = {tp})** → productos correctamente identificados como de baja rotación.  
    - **Falsos positivos (FP = {fp})** → productos mal clasificados como baja rotación (error tipo I).  
    - **Falsos negativos (FN = {fn})** → productos que eran de baja rotación pero el modelo no los detectó (error tipo II).  
    - **Verdaderos negativos (TN = {tn})** → productos correctamente clasificados como de rotación normal.  
    """)

    st.info(f"📊 La precisión total del modelo es del **{precision*100:.2f}%**, lo que significa que predice correctamente aproximadamente {precision*100:.1f}% de los casos analizados.")

    # ================================
    # 5️⃣ Conclusiones gerenciales
    # ================================
    st.markdown("### 💬 Conclusiones Gerenciales")

    interpretaciones = []
    if tp > fn:
        interpretaciones.append("✅ El modelo es bueno identificando productos con baja rotación, lo que permite priorizar promociones o estrategias para esos artículos.")
    else:
        interpretaciones.append("⚠️ El modelo tiene dificultad para detectar productos de baja rotación. Sería recomendable aumentar la variabilidad de datos o incluir más variables como categoría, temporada o cliente.")
    
    if fp > 0.3 * total:
        interpretaciones.append("⚠️ Existen varios falsos positivos: el modelo podría estar señalando productos normales como 'baja rotación', lo que podría llevar a promociones innecesarias.")
    
    if acc > 0.8:
        interpretaciones.append("💪 El modelo muestra un buen nivel de precisión, adecuado para apoyar decisiones comerciales de reposición o liquidación de stock.")
    elif acc > 0.6:
        interpretaciones.append("🟠 El modelo tiene una precisión moderada. Puede servir como referencia inicial, pero conviene optimizarlo con más variables o técnicas avanzadas.")
    else:
        interpretaciones.append("🔴 El modelo aún no tiene precisión suficiente para una toma de decisiones confiable. Se recomienda ajustar la proporción de clases o ampliar el dataset.")

    for i in interpretaciones:
        st.markdown(f"- {i}")

    # ================================
    # 6️⃣ Recomendaciones para el negocio
    # ================================
    st.markdown("### 💡 Recomendaciones para Aurelion")

    st.markdown("""
    - **Productos identificados con baja rotación (1)**: planificar estrategias de promoción, combos o descuentos para acelerar su salida.
    - **Productos de rotación normal (0)**: mantener niveles de inventario estables.
    - **Falsos negativos** (productos lentos no detectados) pueden generar sobrestock; se recomienda incluir más variables (por ejemplo, categoría o frecuencia de venta).
    - Incorporar en el futuro variables como: tipo de cliente, canal de venta, región o temporada, para mejorar la capacidad predictiva.
    - Evaluar modelos no lineales (por ejemplo, RandomForestClassifier) para capturar relaciones más complejas.
    """)

    return modelo, X_test, y_test, y_pred



# ============================================================
# 5️⃣ FUNCIÓN PRINCIPAL
# ============================================================

def mostrar_prediccion_ventas_view(datasets):
    """Vista principal del módulo de predicción de ventas y clasificación."""
    st.title("📊 Predicción y Clasificación de Ventas - Aurelion")

    # Submenú interno
    submenu = st.radio(
        "Selecciona una sección:",
        ["Ampliación del dataset", "Predicción (Regresión)", "Clasificación (Matriz de Confusión)"],
        horizontal=True
    )

    if submenu == "Ampliación del dataset":
        mostrar_ampliacion_dataset(datasets)
    elif submenu == "Predicción (Regresión)":
        # Si hay dataset ampliado en memoria, usarlo
        if st.session_state["dataset_ampliado"] is not None:
            st.info("📦 Usando dataset ampliado almacenado en memoria.")
            df = st.session_state["dataset_ampliado"]
        else:
            st.warning("⚠️ No hay dataset ampliado en memoria. Se usará el dataset base.")
            df = preparar_datos_regresion(datasets)

        if not df.empty:
            comparar_regresiones(df)

    elif submenu == "Clasificación (Matriz de Confusión)":
        if st.session_state["dataset_ampliado"] is not None:
            st.info("📦 Usando dataset ampliado almacenado en memoria.")
            df = st.session_state["dataset_ampliado"]
        else:
            st.warning("⚠️ No hay dataset ampliado en memoria. Se usará el dataset base.")
            df = preparar_datos_regresion(datasets)

        if not df.empty:
            mostrar_matriz_confusion(df)

