# Proyecto Aurelion

# 🛍️ Mi Marketplace - Proyecto de Estudio de IA y Análisis de Datos con Excel

Este proyecto Python es un entorno de estudio integral que simula la gestión de un marketplace. Se centra en la aplicación práctica de conceptos de Inteligencia Artificial y Machine Learning, utilizando `pandas` y `numpy` para la manipulación y análisis de datos almacenados en archivos Excel, y culminando en la visualización de insights mediante un dashboard en Power BI.

## 🎯 Objetivo del Proyecto

El objetivo principal de este proyecto es:
*   **Dominar el ciclo de vida del análisis de datos:** Desde la limpieza y transformación de datos hasta el modelado de Machine Learning y la visualización de resultados.
*   **Aplicar Python y librerías clave:** Utilizar `pandas` para gestión de datos, `numpy` para operaciones numéricas eficientes, y `scikit-learn` para implementar modelos de Machine Learning.
*   **Generar insights accionables:** Obtener conclusiones significativas de los datos del marketplace y presentarlas de manera efectiva en un dashboard de Power BI.
*   **Simular un entorno de marketplace:** Gestionar información de clientes, productos y transacciones de ventas con archivos Excel.

## ✨ Características y Archivos de Datos

El proyecto se basa en los siguientes archivos de Excel, que actúan como nuestra "base de datos" para las distintas fases del análisis:

*   **`clientes.xlsx`**: Contiene la información detallada de los clientes registrados en el marketplace.
    *   **Columnas:** `id_cliente`, `nombre_cliente`, `email`, `ciudad`, `fecha_alta`.
*   **`productos.xlsx`**: Almacena los datos de los productos disponibles para la venta.
    *   **Columnas:** `id_producto`, `nombre_producto`, `categoria`, `precio_unitario`.
*   **`ventas.xlsx`**: Contiene el resumen de cada transacción de venta realizada en el marketplace.
    *   **Columnas:** `id_venta`, `fecha`, `id_cliente`, `nombre_cliente`, `email`, `medio_pago`.
*   **`detalle_ventas.xlsx`**: Registra los productos individuales que forman parte de cada venta, con sus cantidades y precios específicos.
    *   **Columnas:** `id_venta`, `id_producto`, `nombre_producto`, `cantidad`, `precio_unitario`, `importe`.

**Funcionalidades del proyecto:**

1.  **Análisis y Preprocesamiento de Datos con Python:**
    *   **Limpieza y Transformación:** Lectura, inspección y limpieza de datos de los archivos Excel usando `pandas`.
    *   **Estadística Aplicada:** Análisis descriptivo, distribuciones de datos y correlaciones utilizando `pandas` y `numpy`.
    *   **Visualización en Python:** Creación de gráficos exploratorios con `Matplotlib` y `Seaborn`.
2.  **Machine Learning con scikit-learn:**
    *   **Modelado:** Implementación de algoritmos de ML, preparación de datos (división train/test), entrenamiento y evaluación de modelos predictivos.
3.  **Visualización y Reportes con Power BI:**
    *   **Dashboard Interactivo:** Creación de un informe final y un dashboard interactivo en Power BI, cargando los datos procesados o directamente desde los Excel, estableciendo un modelo de datos robusto (tablas, relaciones, DAX).

## 📁 Estructura del Proyecto

```text
.
├── README.md
├── main.py                # Script principal con la lógica de gestión y análisis
├── src/
│   ├── __init__.py
│   ├── data_manager.py    # Funciones para leer/escribir en los archivos Excel
│   ├── models.py          # Definición de estructuras de datos/clases (e.g., Producto, Cliente, Venta)
│   └── utils.py           # Funciones de utilidad auxiliares para limpieza, cálculo, etc.
├── data/
│   ├── clientes.xlsx      # Información detallada de clientes
│   ├── detalle_ventas.xlsx# Detalles de cada producto en una venta
│   ├── productos.xlsx     # Catálogo de productos
│   └── ventas.xlsx        # Resumen de cada transacción de venta
└── requirements.txt       # Dependencias del proyecto
```

## 🛠️ Tecnologías Utilizadas

*   **Python 3.x**
*   **pandas**: Biblioteca esencial para la lectura, manipulación, análisis y escritura de datos tabulares en archivos Excel.
*   **openpyxl**: Backend necesario para `pandas` para interactuar con archivos de Excel en formato `.xlsx`.
*   **scikit-learn**: Implementación de algoritmos de `Machine Learning`.
*   **Matplotlib y Seaborn**: `Visualización` de datos en Python.
*   **Power BI Desktop**: Creación de informes y `dashboards interactivos`.

## 🚀 Puesta en Marcha

Sigue estos pasos para configurar y ejecutar el proyecto en tu entorno local:

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd mi-marketplace-python

```
## 2. Crear y activar un entorno virtual
```bash

python -m venv venv
# Activar el entorno virtual
# En Windows (CMD/PowerShell):
.\venv\Scripts\activate
# En macOS/Linux (Bash/Zsh):
source venv/bin/activate
```
## 3. Instalar dependencias
```bash

pip install -r requirements.txt
```
Asegúrate de que `requirements.txt` contenga las siguientes líneas:
```bash
pandas
numpy
openpyxl
scikit-learn
matplotlib
seaborn
```

## 4. Preparar los archivos de datos

Asegúrate de que los cuatro archivos de Excel (`clientes.xlsx`, `detalle_ventas.xlsx`, `productos.xlsx`, `ventas.xlsx`) estén ubicados en la carpeta `data/` del proyecto. Estos archivos son la fuente de datos para el marketplace.

## 5. Ejecutar el script principal
```bash

python main.py
```
(Aquí puedes añadir instrucciones sobre cómo interactuar con tu `main.py`, por ejemplo, si presenta un menú de opciones en consola.)

## 📝 Uso
El main.py probablemente centraliza la lógica principal del marketplace. A través de este, podrás:

* Cargar y preprocesar los datos de todos los archivos Excel.
* Realizar análisis exploratorios y estadísticos.
* Implementar y evaluar modelos de Machine Learning.
* (Añade ejemplos específicos de funcionalidades implementadas: "generar reportes de ventas", "predecir la demanda de productos", etc.)

Ejemplo básico de cómo cargar datos en `main.py` usando `data_manager.py`:

# En main.py
```bash
python

# En main.py
from src.data_manager import cargar_dataframe_excel

# Cargar los DataFrames al inicio
df_clientes = cargar_dataframe_excel('clientes.xlsx')
df_productos = cargar_dataframe_excel('productos.xlsx')
df_ventas = cargar_dataframe_excel('ventas.xlsx')
df_detalle_ventas = cargar_dataframe_excel('detalle_ventas.xlsx')

print("¡Todos los datos de tu marketplace han sido cargados exitosamente para su estudio!")
# Ahora puedes operar con estos DataFrames para la fase de análisis y ML.
```
**Ahora puedes operar con estos DataFrames para gestionar tu marketplace.**

**Desarrollado por: Edwar Jaramillo**
**Contacto: [Perfil github](https://github.com/eajaramillo)**