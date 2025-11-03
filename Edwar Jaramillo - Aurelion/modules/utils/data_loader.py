import pandas as pd

#Maneja carga y guardado de datasets centralizado.
def cargar_excel(path):
    try:
        return pd.read_excel(path)
    except Exception as e:
        raise FileNotFoundError(f"No se pudo cargar el archivo {path}: {e}")

def guardar_excel(df, path):
    try:
        df.to_excel(path, index=False)
    except Exception as e:
        raise IOError(f"Error al guardar el archivo {path}: {e}")
