import streamlit as st
import pandas as pd
import plotly.express as px

# --- Configuración general ---
st.set_page_config(page_title="Rendimiento E-commerce", page_icon="📈", layout="wide")

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

# --- Filtros ---
meses = st.sidebar.multiselect("Selecciona los meses:", options=df['Mes'], default=df['Mes'])
df_filtrado = df[df['Mes'].isin(meses)]

# --- Métricas destacadas ---
st.title("📊 Dashboard de Rendimiento E-commerce")
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Ventas", f"${df_filtrado['Ventas'].sum():,.0f}")
col2.metric("📢 Gasto Publicidad", f"${df_filtrado['Gasto_Publicidad'].sum():,.0f}")
col3.metric("🎯 Conversión Promedio", f"{df_filtrado['Conversion (%)'].mean():.2f}%")
col4.metric("📈 Eficiencia Media", f"{df_filtrado['Eficiencia'].mean():.2f}")

# --- Gráfico 1: Ventas vs Publicidad ---
fig1 = px.bar(
    df_filtrado,
    x='Mes', y=['Ventas', 'Gasto_Publicidad'],
    barmode='group',
    title='📊 Ventas vs Gasto en Publicidad',
    labels={'value': 'Monto ($)', 'variable': 'Categoría'},
    color_discrete_sequence=['#1f77b4', '#ff7f0e'],
    text_auto=True
)
st.plotly_chart(fig1, use_container_width=True)

# --- Gráfico 2: Conversión y Eficiencia ---
fig2 = px.line(
    df_filtrado,
    x='Mes', y=['Conversion (%)', 'Eficiencia'],
    markers=True,
    title='📈 Conversión (%) y Eficiencia (Ventas/Gasto)',
    color_discrete_sequence=['green', 'royalblue']
)
fig2.update_traces(line=dict(width=3))
st.plotly_chart(fig2, use_container_width=True)

# --- Gráfico 3: Ticket promedio y ROI ---
fig3 = px.bar(
    df_filtrado,
    x='Mes', y=['Ticket_Promedio', 'ROI (%)'],
    barmode='group',
    title='💰 Ticket Promedio y ROI (%) por Mes',
    labels={'value': 'Valor', 'variable': 'Indicador'},
    color_discrete_sequence=['#636EFA', '#EF553B'],
    text_auto=True
)
st.plotly_chart(fig3, use_container_width=True)

# --- Tabla final ---
st.subheader("📋 Datos base filtrados")
st.dataframe(df_filtrado)
