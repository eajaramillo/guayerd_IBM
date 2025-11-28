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
    """Vista interactiva para ampliar el dataset con ventas únicas, guardar en memoria y exportar."""
    st.subheader("🧩 Ampliación del Dataset con Ventas Únicas")

    # 1️⃣ Construir la tabla maestra base
    df = construir_tabla_maestra(datasets, mostrar_mensajes=False)
    if df.empty:
        st.warning("⚠️ No hay datos disponibles para ampliar.")
        return

    # Validaciones básicas
    if "id_venta" not in df.columns or "cantidad" not in df.columns:
        st.error("❌ La tabla maestra debe tener las columnas 'id_venta' y 'cantidad'.")
        return

    # Convertir fecha a tipo datetime
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    # Variables base
    base_ventas = df["id_venta"].nunique()
    base_productos = df["nombre_producto"].nunique() if "nombre_producto" in df.columns else 0
    base_categorias = df["categoria"].nunique() if "categoria" in df.columns else 0
    base_cantidades = df["cantidad"].sum() if "cantidad" in df.columns else 0

    # 2️⃣ Slider de factor de ampliación
    factor = st.slider("Multiplicar dataset por:", 1, 10, 1)
    st.markdown("Aumenta el tamaño del dataset para simular nuevas ventas y mejorar el entrenamiento del modelo.")

    # 3️⃣ Generar dataset ampliado con ventas únicas
    df_extendido = []
    for i in range(factor):
        df_copy = df.copy()

        # Regenerar IDs únicos de venta
        df_copy["id_venta"] = df_copy["id_venta"].astype(str) + f"_{i+1}"

        # Desplazar fechas (simulación mensual)
        if "fecha" in df_copy.columns:
            df_copy["fecha"] = df_copy["fecha"] + timedelta(days=30 * i)

        # Variar cantidades ±10%
        df_copy["cantidad"] = (df_copy["cantidad"] * np.random.uniform(0.9, 1.1, len(df_copy))).round().astype(int)
        df_copy["cantidad"] = df_copy["cantidad"].clip(lower=1)

        # Variar importe_total ±5%
        if "importe_total" in df_copy.columns:
            df_copy["importe_total"] = (df_copy["importe_total"] * np.random.uniform(0.95, 1.05, len(df_copy))).round(2)

        df_extendido.append(df_copy)

    df_extendido = pd.concat(df_extendido, ignore_index=True)

    # 4️⃣ Calcular nuevas métricas
    nuevas_ventas = df_extendido["id_venta"].nunique()
    nuevas_cantidades = df_extendido["cantidad"].sum()
    nuevas_categorias = df_extendido["categoria"].nunique()
    nuevos_productos = df_extendido["nombre_producto"].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧾 Ventas únicas", f"{nuevas_ventas:,}", f"x{factor}")
    col2.metric("📦 Productos", f"{nuevos_productos:,}")
    col3.metric("🏷️ Categorías", f"{nuevas_categorias:,}")
    col4.metric("📊 Total de cantidades", f"{nuevas_cantidades:,}")

    # 5️⃣ Visualización del crecimiento
    data_crecimiento = pd.DataFrame({
        "Factor": list(range(1, factor + 1)),
        "Ventas Totales": [base_ventas * i for i in range(1, factor + 1)],
        "Cantidad Total": [base_cantidades * i for i in range(1, factor + 1)]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data_crecimiento["Factor"],
        y=data_crecimiento["Ventas Totales"],
        mode="lines+markers",
        name="Ventas Totales",
        line=dict(color="#00BFFF", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=data_crecimiento["Factor"],
        y=data_crecimiento["Cantidad Total"],
        mode="lines+markers",
        name="Cantidad Total",
        line=dict(color="#32CD32", width=3)
    ))
    fig.update_layout(
        title="📈 Crecimiento del Dataset Ampliado",
        xaxis_title="Factor de Ampliación",
        yaxis_title="Totales",
        legend_title="Métricas",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 6️⃣ Guardar en memoria para las demás secciones
    st.session_state["dataset_ampliado"] = df_extendido
    st.success(f"✅ Dataset ampliado ({len(df_extendido)} registros) guardado en memoria global.")
    st.info("💡 Ahora puedes ir a las secciones de *Regresión* o *Clasificación* para usar este dataset ampliado.")


    # 7️⃣ Mostrar vista previa
    st.markdown("### 📄 Vista previa del dataset ampliado")
    st.dataframe(df_extendido.head(10), use_container_width=True)

    # 8️⃣ Botones para exportar dataset
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
# 3️⃣ REGRESIÓN: LINEAL vs KNN
# ============================================================

def comparar_regresiones(df):
    """Compara modelos de regresión lineal y KNN para predecir cantidades."""
    st.subheader("📈 Comparación: LinearRegression vs KNeighborsRegressor")

    X = df[["importe_total", "mes"]]
    y = df["cantidad"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Modelos
    lr = LinearRegression()
    knn = KNeighborsRegressor(n_neighbors=5)
    rfr = RandomForestRegressor()

    lr.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    rfr.fit(X_train, y_train)

    # Predicciones
    y_pred_lr = lr.predict(X_test)
    y_pred_knn = knn.predict(X_test)
    y_pred_rfr = rfr.predict(X_test)

    # Métricas
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)
    
    mse_rfr = mean_squared_error(y_test, y_pred_rfr)
    r2_rfr = r2_score(y_test, y_pred_rfr)

    # Resultados
    st.write(f"**LinearRegression** → MSE: {mse_lr:.2f} | R²: {r2_lr:.3f}")
    st.write(f"**KNeighborsRegressor** → MSE: {mse_knn:.2f} | R²: {r2_knn:.3f}")
    st.write(f"**RandomForestRegressor** → MSE: {mse_rfr:.2f} | R²: {r2_rfr:.3f}")

    # Curva de predicción comparativa
    X_grid = np.linspace(X["importe_total"].min(), X["importe_total"].max(), 200).reshape(-1, 1)
    X_grid_full = np.hstack((X_grid, np.full_like(X_grid, X["mes"].mean())))

    y_grid_lr = lr.predict(X_grid_full)
    y_grid_knn = knn.predict(X_grid_full)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train["importe_total"], y_train, label="Train (puntos)", alpha=0.6)
    plt.scatter(X_test["importe_total"], y_test, label="Test (puntos)", marker='x')
    plt.plot(X_grid, y_grid_lr, label="LinearRegression (pred)", linewidth=2)
    plt.plot(X_grid, y_grid_knn, label="KNNRegressor (pred)", linewidth=2)
    plt.xlabel("Importe Total")
    plt.ylabel("Cantidad Vendida")
    plt.title("Regresión: LinearRegression vs KNeighborsRegressor")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    return lr, knn, X_test, y_test, y_pred_lr, y_pred_knn


# ============================================================
# 4️⃣ CLASIFICACIÓN Y MATRIZ DE CONFUSIÓN
# ============================================================

def mostrar_matriz_confusion(df):
    """Entrena un modelo de clasificación (baja rotación) y muestra matriz de confusión."""
    st.subheader("🔍 Análisis de clasificación: Matriz de Confusión")

    # Simular columna binaria
    threshold = df["cantidad"].median()
    df["baja_rotacion"] = df["cantidad"] < threshold

    X = df[["importe_total", "mes"]]
    y = df["baja_rotacion"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    st.metric("🎯 Precisión del modelo", f"{acc*100:.2f}%")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="viridis")
    plt.title("Matriz de Confusión - Baja Rotación")
    st.pyplot(plt)

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

