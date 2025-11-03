import pandas as pd
import numpy as np

#Funciones reutilizables de limpieza.
def eliminar_duplicados(df):
    return df.drop_duplicates()

def eliminar_nulos(df):
    return df.dropna()

def rellenar_nulos(df, metodo="0"):
    if metodo == "0":
        return df.fillna(0)
    elif metodo == "mediana":
        return df.fillna(df.median(numeric_only=True))
    return df
