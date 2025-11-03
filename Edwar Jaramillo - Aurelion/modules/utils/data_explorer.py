import pandas as pd
import io

"""
Esta clase es extensible: si luego quieres añadir nuevas operaciones (corr, dtypes, value_counts, etc.), solo agregas un elif dentro de ejecutar_accion.
"""
class DataExplorer:
    """Clase responsable de ejecutar acciones de exploración sobre DataFrames."""

    def __init__(self, datasets: dict):
        """
        datasets: diccionario con nombre_tabla -> DataFrame
        """
        self.datasets = datasets

    def ejecutar_accion(self, df: pd.DataFrame, accion: str):
        """Ejecuta una acción específica sobre un DataFrame y devuelve su resultado."""
        if accion == "describe":
            return df.describe(include="all")
        elif accion == "info":
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            return pd.DataFrame({"info": info_str.splitlines()})
        elif accion == "head":
            return df.head()
        elif accion == "shape":
            return pd.DataFrame(
                {"Filas": [df.shape[0]], "Columnas": [df.shape[1]]}
            )
        elif accion == "columns":
            return pd.DataFrame({"Columnas": df.columns})
        else:
            return pd.DataFrame({"Resultado": [f"Acción '{accion}' no implementada."]})

    def explorar(self, tablas: list, acciones: list):
        """Ejecuta varias acciones sobre varias tablas."""
        resultados = {}
        for tabla in tablas:
            df = self.datasets.get(tabla)
            if df is None:
                resultados[tabla] = {"Error": f"No se encontró la tabla {tabla}"}
                continue

            resultados[tabla] = {}
            for accion in acciones:
                resultados[tabla][accion] = self.ejecutar_accion(df, accion)
        return resultados
