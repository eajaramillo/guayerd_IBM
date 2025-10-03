# Proyecto Aurelion

# 🛍️ Mi Marketplace - Gestión de Datos con Excel

Este proyecto Python simula la gestión de un marketplace utilizando archivos de Excel como una base de datos simple y fácil de entender, almacenando información esencial de clientes, productos y transacciones de ventas.

## 🎯 Objetivo del Proyecto

El objetivo principal de este proyecto es:
*   Demostrar el manejo eficiente de datos tabulares en Python, haciendo uso extensivo de la librería `pandas` para interactuar con archivos `.xlsx`.
*   Simular operaciones clave de un marketplace, incluyendo la consulta y gestión de clientes, productos, y el procesamiento detallado de ventas.
*   Proporcionar un punto de partida claro y funcional para el desarrollo de funcionalidades más avanzadas en un entorno de gestión de marketplace.

## ✨ Características y Archivos de Datos

El proyecto se basa en los siguientes archivos de Excel, que actúan como nuestra "base de datos":

*   **`clientes.xlsx`**: Contiene la información detallada de los clientes registrados en el marketplace.
    *   **Columnas:** `id_cliente`, `nombre_cliente`, `email`, `ciudad`, `fecha_alta`.
*   **`productos.xlsx`**: Almacena los datos de los productos disponibles para la venta.
    *   **Columnas:** `id_producto`, `nombre_producto`, `categoria`, `precio_unitario`.
*   **`ventas.xlsx`**: **Contiene el resumen de cada transacción de venta realizada en el marketplace.**
    *   **Columnas:** `id_venta`, `fecha`, `id_cliente`, `nombre_cliente`, `email`, `medio_pago`.
*   **`detalle_ventas.xlsx`**: Registra los productos individuales que forman parte de cada venta, con sus cantidades y precios específicos.
    *   **Columnas:** `id_venta`, `id_producto`, `nombre_producto`, `cantidad`, `precio_unitario`, `importe`.

**Funcionalidades del proyecto:**

*   **Gestión de Clientes:** Carga, consulta, filtrado y potencialmente actualización de la información de `clientes.xlsx`.
*   **Gestión de Productos:** Carga, consulta de `productos.xlsx` y, potencialmente, lógica para actualizar el stock (si se añade la columna en `productos.xlsx`).
*   **Procesamiento de Ventas:** Lógica para registrar nuevas ventas, vincularlas a clientes y productos, y gestionar los medios de pago en `ventas.xlsx` y `detalle_ventas.xlsx`.
*   **Análisis Detallado de Ventas:** Consulta y análisis de `detalle_ventas.xlsx` para entender el desglose de productos en cada transacción y `ventas.xlsx` para los resúmenes.
*   **Reportes y Métricas:** Posibilidad de generar informes sobre ventas totales, productos más vendidos, clientes top, medios de pago preferidos, y tendencias de ventas.

## 📁 Estructura del Proyecto

├── README.md
├── main.py # Script principal con la lógica del marketplace
├── src/
│ ├── init.py
│ ├── data_manager.py # Funciones para leer/escribir en los archivos Excel
│ ├── models.py # Definición de estructuras de datos/clases (e.g., Producto, Cliente, Venta)
│ └── utils.py # Funciones de utilidad auxiliares
├── data/
│ ├── clientes.xlsx # Información detallada de clientes
│ ├── detalle_ventas.xlsx# Detalles de cada producto en una venta
│ ├── productos.xlsx # Catálogo de productos
│ └── ventas.xlsx # Resumen de cada transacción de venta
└── requirements.txt # Dependencias del proyecto

## 🛠️ Tecnologías Utilizadas

*   **Python 3.x**
*   **pandas**: Biblioteca esencial para la lectura, manipulación, análisis y escritura de datos tabulares en archivos Excel.
*   **openpyxl**: Backend necesario para `pandas` para interactuar con archivos de Excel en formato `.xlsx`.

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
## 4. Preparar los archivos de datos

Asegúrate de que los cuatro archivos de Excel (clientes.xlsx, detalle_ventas.xlsx, productos.xlsx, ventas.xlsx) estén ubicados en la carpeta data/ del proyecto. Estos archivos son la fuente de datos para el marketplace.

## 5. Ejecutar el script principal
```bash

python main.py
```
(Aquí puedes añadir instrucciones sobre cómo interactuar con tu main.py, por ejemplo, si presenta un menú de opciones en consola.)

## 📝 Uso
El main.py probablemente centraliza la lógica principal del marketplace. A través de este, podrás:

    Cargar todos los datos de los archivos Excel al inicio del programa.
    Realizar consultas detalladas sobre clientes, productos o ventas.
    (Añade más funcionalidades específicas que tu main.py permite, por ejemplo, "añadir un nuevo cliente", "registrar una nueva venta", "generar un reporte de ventas por medio de pago", etc.)

Ejemplo básico de cómo cargar datos en main.py usando data_manager.py:

# En main.py
```bash
from src.data_manager import cargar_dataframe_excel
```
# Cargar los DataFrames al inicio
```bash
df_clientes = cargar_dataframe_excel('clientes.xlsx')
df_productos = cargar_dataframe_excel('productos.xlsx')
df_ventas = cargar_dataframe_excel('ventas.xlsx')
df_detalle_ventas = cargar_dataframe_excel('detalle_ventas.xlsx')

print("¡Todos los datos de tu marketplace han sido cargados exitosamente!")
```
# Ahora puedes operar con estos DataFrames para gestionar tu marketplace.

Desarrollado por: Edwar Jaramillo
Contacto: [Perfil github](https://github.com/eajaramillo)