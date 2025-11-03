import seaborn as sns
import matplotlib.pyplot as plt

#Centraliza gráficos básicos.
def grafico_boxplot(df, columna):
    fig, ax = plt.subplots()
    sns.boxplot(x=df[columna], ax=ax)
    return fig

def grafico_histograma(df, columna):
    fig, ax = plt.subplots()
    sns.histplot(df[columna], kde=True, ax=ax)
    return fig

def grafico_heatmap(df):
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    return fig
