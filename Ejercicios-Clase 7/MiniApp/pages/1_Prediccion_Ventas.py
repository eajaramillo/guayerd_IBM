import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --- Configuración ---
st.set_page_config(page_title="Predicción de Ventas", page_icon="🤖", layout="wide")

st.title("📈 Predicción de Ventas - Machine Learning")

# --- Datos base ---
data = {
    'Mes': ['Ene', 'Feb', 'Mar', 'Abr', 'May'],
    'Ventas': [45000, 52000, 38000, 61000, 48000],
    'Visitantes': [15000, 18200, 12500, 20500, 16800],
    'Conversion (%)': [3.2, 2.9, 3.8, 3.1, 2.7],
    'Gasto_Publicidad': [8500, 9800, 7200, 11200, 9500],
    'Productos_Vendidos': [450, 520, 380, 610, 480]
}

df = pd.DataFrame(data)
df['Eficiencia'] = df['Ventas'] / df['Gasto_Publicidad']
df['Ticket_Promedio'] = df['Ventas'] / df['Productos_Vendidos']
df['ROI (%)'] = ((df['Ventas'] - df['Gasto_Publicidad']) / df['Gasto_Publicidad']) * 100

# --- Correlaciones ---
st.subheader("🔍 Análisis de correlaciones")

corr = df[['Ventas', 'Visitantes', 'Gasto_Publicidad', 'Conversion (%)', 'Eficiencia']].corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Matriz de Correlación")
st.plotly_chart(fig_corr, use_container_width=True)

# --- Selección de variables para entrenamiento ---
st.subheader("⚙️ Entrenamiento del modelo de predicción")

x_var = st.selectbox("Selecciona la variable predictora (X):", ['Visitantes', 'Gasto_Publicidad', 'Conversion (%)', 'Eficiencia'])
X = df[[x_var]]
y = df['Ventas']

# División en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# --- Evaluación ---
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("📊 MAE (Error Absoluto Medio)", f"${mae:,.0f}")
col2.metric("📈 R² (Bondad de ajuste)", f"{r2:.2f}")

# --- Gráfico de regresión ---
fig_reg = px.scatter(
    df,
    x=x_var, y='Ventas',
    trendline='ols',
    title=f"Relación entre {x_var} y Ventas (Modelo Lineal)"
)
st.plotly_chart(fig_reg, use_container_width=True)

# --- Predicción personalizada ---
st.subheader("🔮 Probar una predicción manual")
valor = st.number_input(f"Ingrese un valor de {x_var}:", min_value=0.0)
if valor > 0:
    prediccion = modelo.predict([[valor]])[0]
    st.success(f"Predicción de Ventas: **${prediccion:,.0f}** con {x_var} = {valor:,.0f}")
