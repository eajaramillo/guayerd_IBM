# 1. Documentación proyecto Aurelion
___

## 1. Tema
**🛍️ Mi Minimarket - Proyecto de Estudio de IA y Análisis de Datos con Excel**

Este proyecto Python es un entorno de estudio integral que simula la gestión de un minimarket. Se centra en la aplicación práctica de conceptos de Inteligencia Artificial y Machine Learning, utilizando librerías, tecnologías y conceptos como:

**Tecnologías y librerías**
```
* Análisis estadístico
* Tecnologías y librerías VSCode y plugins
* Lenguaje Python
* Pandas
* numpy
* openpyxl
* streamlit
* plotly
* matplotlib
* python-dateutil
* Jupiter
* PowerBI
```

**Conceptos**
```
* Limpieza de datos
* Correlacionea
* ETLs
* Lectura de archivos
* Estructuras principales
* Inspección y limpieza
* Estadística descriptiva básica
* Distribuciones de datos
* Correlaciones
* Visualización - Matplotlib - Seaborn
* Machine Learning
* Tipos de aprendizajes
* Algoritmos básicos
* Métricas de evaluación
* Preparación datos
* División train/test
* Proceso entrenamiento
* Evaluación modelos
* Algoritmos específicos
```

Con esta tecnoloigías y conceptos para la manipulación y análisis de datos almacenados en archivos Excel, se busca realizar estudios y encontrar datos concluyentes que puedan ser usados para exponer a las áreas gerenciales culminando en la visualización de insights mediante un dashboard en Power BI que permitan sustentar los hallazgos.

___

## 2. 🎯 Problema - Objetivo del Proyecto

El objetivo principal de este proyecto es:
*   **Dominar el ciclo de vida del análisis de datos:** Desde la limpieza y transformación de datos hasta el modelado de Machine Learning y la visualización de resultados.
*   **Aplicar Python y librerías clave:** Utilizar `pandas` para gestión de datos, `numpy` para operaciones numéricas eficientes, y `scikit-learn` para implementar modelos de Machine Learning. Adicionalmente incorporar todas las demás tecnologías descritos en el *[Tema](#tema)* de este proyecto
*   **Generar insights accionables:** Obtener conclusiones significativas de los datos del marketplace y presentarlas de manera efectiva en un dashboard de Power BI.
*   **Simular un entorno de marketplace:** Gestionar información de clientes, productos y transacciones de ventas con archivos Excel.


### 🚀 OBJETIVO complementario

* Construir una aplicación web en Streamlit que permita:
* Consultar la documentación dinámica (Markdown).
* Gestionar registros del minimarket (clientes, productos, ventas).
* Analizar y limpiar los datasets.
* Aplicar análisis estadístico y visualización interactiva.
* Evolucionar a Machine Learning más adelante.

___

## 3. Solución

Se propone la elaboración de una **documentación detallada** y estructurada para para este proyecto de tal manera que describa los procesos, componentes y funcionalidades que se plasmarán en un programa ejecutable. Esta documentación servirá como base para el diseño, implementación y mantenimiento de futuras soluciones tecnológicas dentro de esta solución.

Adicionalmente, se plantea la creación de un **programa** que permita visualizar la documentación desde un menú de **forma interactiva**, facilitando la consulta de la documentación completa el proyecto. Este sistema interactivo permitirá acceder a la información técnica de manera dinámica, organizada y actualizable; promoviendo la colaboración y mejora continua de los procesos tecnológicos del proyecto. Este programa interactivo también estará en la capacidad de ejecutar y mostrar en análisis, pasos y procesos que llevarán a la conclusión final luego del estudio de los datos analizados, mediante submenús intermedios que alimentando la base de datos y entregables para la solución definitiva.

___

## 4. Dataset de referencia: fuente, definición, estructura, tipos y escala de medición
# 4.1 Fuente
El proyecto se basa en los siguientes archivos de Excel, que actúan como nuestra "base de datos" para las distintas fases del análisis. Esta base de datos contiene la información inicial de la muestra, sin embargo puede sufrir transformaciones luego de la limpieza de datos o inyección de nueva data para ampliar el rango de la muestra.

# 4.2 Definición
**Simular un entorno de marketplace:** Gestionar información de clientes, productos y transacciones de ventas con archivos Excel.

**Clientes** 
*   **`clientes.xlsx`**  — ~100 filas: Contiene la información detallada de los clientes registrados en el marketplace.
**Detalle de campos**

| Campo             | Tipo | Escala   |
|-------------------|------|----------|
| id_cliente        | int  | Nominal  |
| nombre_cliente    | str  | Nominal  |
| email             | str  | Nominal  |
| ciudad            | str  | Nominal  |
| fecha_alta        | date | Intervalo|

**Productos**
*   **`productos.xlsx`** — ~100 filas: Almacena los datos de los productos disponibles para la venta.
**Detalle de campos**

| Campo             | Tipo | Escala   | 
|-------------------|------|----------| 
| id_producto       | int  | Nominal  | 
| nombre_producto   | str  | Nominal  |
| categoria         | str  | Nominal  |
| precio_unitario   | int  | Razón    |

**Ventas**
*   **`ventas.xlsx`** — ~120 filas: Contiene el resumen de cada transacción de venta realizada en el marketplace.
**Detalle de campos**

| Campo             | Tipo | Escala    |
|-------------------|------|-----------|
| id_venta          | int  | Nominal   |
| fecha             | date | Intervalo |
| id_cliente        | int  | Nominal   |
| nombre_cliente    | str  | Nominal   |
| email             | str  | Nominal   |
| medio_pago        | str  | Nominal   |

**Detalle Ventas**
*   **`detalle_ventas.xlsx`** — ~300 filas: Registra los productos individuales que forman parte de cada venta, con sus cantidades y precios específicos.
**Detalle de campos**

| Campo             | Tipo | Escala  |
|-------------------|------|---------|
| id_venta          | int  | Nominal |
| id_producto       | int  | Nominal |
| nombre_producto   | str  | Nominal |
| cantidad          | int  | Razón   |
| precio_unitario   | int  | Razón   |
| importe           | int  | Razón   |

___

## 5. Pasos
1. Abrir o correr el programa por consola
2. Mostrar el menú de opciones
3. Leer las opciones
4. Permitir que el usuario escriba una opción del menú
5. Enviar la opción seleccionada
6. Mostrar la información que corresponde a la opción seleccionada
7. Mostrar la opción de cerrar o elegir otra opción
8. Ejecutar la opción que corresponda, resolviendo la petición o cerrando el programa

___

## 6. Pseudocódigo 
Pseudocódigo o flujo de ejecución dentro del programama en python.

```
Algoritmo MiMinimarket
	Escribir "Bienvenido a Mi Minimarket"
	Definir opcion_menu Como Entero
	opcion_menu = 1
	Mientras opcion_menu <> 0 Hacer
		Escribir "Seleccione una de las opciones"
		Escribir "Opciones"
		Escribir "1. Tema"
		Escribir "2. Problema"
		Escribir "3. Solución"
		Escribir "4. Caracteristicas set de datos"
		Escribir "5. Pasos"
		Escribir "6. Pseudocódigo"
		Escribir "7. Diagrama de flujo"
		Escribir "8. Ejecutar el programa"
		Escribir "9. Sugerencias y mejoras aplicadas con Copilot"
		Escribir "0. Para salir"
		Leer opcion_menu
		Segun opcion_menu Hacer
			1:
				Escribir "Tema"
			2:
				Escribir "Problema"
			3:
				Escribir "Solución"
			4:
				Escribir "Caracteristicas set de datos"
			5:
				Escribir "Los pasos"
			6:
				Escribir "El pseudocódigo"
			7:
				Escribir "El diagrama de flujo"
			8:
				Escribir "Ejecutar el programa"
			9:
				Escribir "Sugerencias realizadas por copilot"
			0:
				Escribir "Gracias por su atención"
			De Otro Modo:
				Escribir "Por favor ingrese una opción válida"
		Fin Segun
	Fin Mientras
FinAlgoritmo
```
___

## 7. Diagrama de flujo

El diagrama de flujo de **Mi Minimarket** se presenta a continuación:

![Ver diagrama](sources/images/diagrama_minimarket.png)

___

## 8. Ejecutar el programa
Para ejecutar el programa se debe abrir por terminal el main.py o sí abre el archivo desde visual studio code, se puede ejecutar directamente desde el play que viene con el ide.

**📁 Estructura del Proyecto V.1**

```text
.
├── README.md                           # Documentación para mostrar en github
├── visor_documentacion_md_buscar.py    # Script principal con la lógica de gestión y análisis
├── sources/imeges
│   ├── diagrama_minimarket.png         # Diagrama de flujo del programa en consola
├── database/
│   ├── clientes.xlsx                   # Información detallada de clientes
│   ├── detalle_ventas.xlsx             # Detalles de cada producto en una venta
│   ├── productos.xlsx                  # Catálogo de productos
│   └── ventas.xlsx                     # Resumen de cada transacción de venta
└── requirements.txt                    # Dependencias del proyecto
```

**📁 Estructura del Proyecto V.2**
```
MiMinimarketApp/
│
├── app.py                            # Punto de entrada principal
│
├── database/                         # Datasets base
│   ├── clientes.xlsx
│   ├── productos.xlsx
│   ├── ventas.xlsx
│   ├── detalle_ventas.xlsx
│
├── modules/                          # Vistas modulares (principio de responsabilidad única)
│   ├── documentacion_view.py         # Módulo Documentación
│   ├── minimarket_view.py            # Módulo de registro de información
│   ├── limpieza_view.py              # Limpieza y transformación de datos
│   ├── estadisticas_view.py          # Análisis y visualización estadística
│   └── utils/
│       ├── data_loader.py            # Carga y guardado centralizado de datasets
│       ├── data_cleaner.py           # Funciones reutilizables de limpieza
│       └── plot_utils.py             # Funciones gráficas comunes
│
├── documentacion.md                  # Archivo dinámico del proyecto
│
└── requirements.txt

```

**🛠️ Tecnologías Utilizadas o a utilizar proximamente**

*   **Python 3.x**
*   **scikit-learn**: Implementación de algoritmos de `Machine Learning`.
*   **Power BI Desktop**: Creación de informes y `dashboards interactivos`.

| Librería          | Propósito                                                             |
| ----------------- | --------------------------------------------------------------------- |
| `streamlit`       | Crear la interfaz web interactiva del proyecto.                       |
| `pandas`          | Manipulación y análisis de datos (lectura, limpieza, agregación). Biblioteca esencial para la lectura, manipulación, análisis y escritura de datos tabulares en archivos Excel.     |
| `numpy`           | Operaciones matemáticas y numéricas.                                  |
| `matplotlib`      | Gráficos base y visualizaciones básicas de datos en Python.                              |
| `seaborn`         | Gráficos estadísticos avanzados (boxplot, heatmap, violinplot).       |
| `openpyxl`        | Permite leer y escribir archivos `.xlsx` (Excel). Backend necesario para `pandas` para interactuar con archivos de Excel en formato `.xlsx`.                    |
| `python-dateutil` | Manejo de fechas y tiempos (útil para las columnas tipo `timestamp`). |
| `plotly`          | Gráficos interactivos opcionales (podrás usarlo en módulos futuros).  |
| `scipy`           | Funciones estadísticas y científicas (para análisis más profundos).   |
| `statsmodels`     | Análisis estadístico avanzado (para futuras clases de IA).            |

___

## 9. Sugerencias y mejoras aplicadas con IA

Main (.py)
* Debe permitir obtener información del proyecto

* **Sugerencias y mejoras aplicadas con IA**
    * Luego de realizar un prompt para mejorar el programa ejecutable, se realizar una separación de la documentación en un diccionario reutilizable y de cierta manera desacoplado.

    * Se realizó una mejora añadiendo dos opciones de menú adicionales que corresponden a:
        * Opción de búsqueda: para localizar palabras clave dentro de la documentación (e.g., “tema”, “solución").
        * Opción de “exportar sección”: para guardar en .txt/.md lo mostrado por pantalla.

___

# 2. Limpieza de datos

En este proceso se realizará un análisis detallado de los datos y sus estructuras en el dataset, se elaborará un menú interactivo que permita revisar las tablas, permitir la modificación individual o masiva de los datos que requiere intervensión para realizar una mejor limpieza de los datos.
___

### 🚀 FASE 1: Configuración y exploración inicial**

:one: Crear la estructura de carpetas (si aún no existe)
:two: Instalar librerías necesarias

**En la terminal**
```
pip install pandas numpy matplotlib seaborn plotly openpyxl jupyter scikit-learn python-dateutil

```

### Mi Minimarket: Análisis de datos inicial**

#### 1. Carga de datos
```
clientes = pd.read_excel('database/clientes.xlsx')
productos = pd.read_excel('database/productos.xlsx')
ventas = pd.read_excel('database/ventas.xlsx')
detalle = pd.read_excel('database/detalle_ventas.xlsx')
```
#### 2. Inspección rápida
```
print("Clientes:")
display(clientes.head())

print("\nProductos:")
display(productos.head())

print("\nVentas:")
display(ventas.head())

print("\nDetalle Ventas:")
display(detalle.head())

```

### 🧹 FASE 2: Limpieza y Transformación (ETL con Pandas)
**🎯 Objetivo**

* Verificar la estructura, dataframes, tipos de datos, valores nulos y duplicados.
* Estandarizar formatos (fechas, texto, numéricos).
* Integrar las 4 tablas (clientes, productos, ventas, detalle_ventas) en una sola vista analítica.

#### Limpieza de datos
**mostrar_valores_faltantes(df)**
* Detecta valores nulos, vacíos o con espacios.
* Muestra las filas afectadas, permite elegir cómo tratarlas y aplicar la acción seleccionada.

**mostrar_duplicados(df)**
* Muestra los registros duplicados y permite decidir si eliminarlos o mantenerlos.

**mostrar_inconsistencias(df)**
* Permite detectar y corregir inconsistencias de formato, incluyendo texto, fechas y tipos.

**Resumen de la limpieza**
| Mejora                            | Descripción                                                      |
| --------------------------------- | ---------------------------------------------------------------- |
| **Vista previa**                  | Se muestran las filas afectadas antes de actuar.                 |
| **Control selectivo**             | Puedes decidir si eliminar, rellenar o corregir.                 |
| **Valores vacíos tratados**       | No solo nulos (`NaN`), también celdas vacías o con espacios.     |
| **Normalización interactiva**     | Se seleccionan columnas específicas y se visualizan los cambios. |
| **Conversión flexible de fechas** | Permite definir formato manualmente.                             |
| **Prevención de errores**         | Cada acción valida y muestra resultados con mensajes claros.     |



#### 3. Inspección de estructura
```
for nombre, df in [('Clientes', clientes), ('Productos', productos), ('Ventas', ventas), ('Detalle Ventas', detalle)]:
    print(f"\n===== {nombre.upper()} =====")
    print(df.info())
    print(f"Duplicados: {df.duplicated().sum()}")
    print(f"Valores nulos:\n{df.isnull().sum()}")

```

#### Recategorización de productos en el dataset

Vamos a realizar un proceso profesional y controlado de recategorización de productos en el dataset productos, de modo que puedas detectar inconsistencias, analizarlas y corregirlas sin perder trazabilidad.

**🔍 Qué observarás:**

* Errores comunes: mayúsculas/minúsculas, tildes, espacios extra, nombres duplicados (ej: “Lácteos”, “lacteos”, “Lacteos”).

* Productos mal ubicados o genéricos (ej: “Sin categoría”, “Otros”, “Default”).

#### Análisis de categorías
```
print("Categorías únicas:")
print(productos['categoria'].unique())

print("\nConteo de productos por categoría:")
display(productos['categoria'].value_counts())

```

#### Normalizar texto antes de recategorizar

Conviene unificar el formato para que no existan variantes de texto:

##### Limpieza de texto en categorías
`productos['categoria'] = productos['categoria'].str.strip().str.title()`

##### Correcciones básicas automáticas
```
productos['categoria'] = productos['categoria'].replace({
    'Lacteos': 'Lácteos',
    'Lacteo': 'Lácteos',
    'Verdura': 'Verduras',
    'Fruta': 'Frutas',
    'Cereal': 'Cereales',
    'Otros': 'Otros Productos',
    'Alimento': 'Alimentos'
})

```

#Bloque automatizado de sugerencias de recategorización

#Este bloque agrupa los productos según coincidencias con palabras clave comunes y te muestra sugerencias para corregirlos.

#### Sugerencias de recategorización automática

**Diccionario de palabras clave → categoría sugerida**

**Opción 1** - *Limitar categorías solo a 'Alimentos' y 'Limpieza'*

```
reglas_categoria = {
    "Jugo|Bebida|Agua|Refresco|Gaseosa|Té|Energética|Mate|Pepsi|Cerveza|Avena|Vino|Ron|Whisky|Fernet": "Alimentos",
    "Pan|Ponqué|Bizcocho|Panela|Mermelada|Manteca": "Alimentos",
    "Yogur|Leche|Queso|Mantequilla": "Alimentos",
    "Arroz|Frijol|Lenteja|Cereal": "Alimentos",
    "Manzana|Banano|Naranja|Pera|Uva": "Alimentos",
    "Tomate|Cebolla|Papa|Lechuga|Zanahoria|Verduras": "Alimentos",
    "Detergente|Jabón|Limpiador|Desinfectante": "Limpieza",
    "Galleta|Chocolate|Dulce|Confite|Alfajor|Maní|Turrón|Azúcar|Caramelo|Chupetín|Stevia|Pizza|Helado|Aceitunas|Caldo": "Alimentos"
}

```
**Opción 2** - *Seperar en nuevas categorías más detalladas*
```
reglas_categoria = {
    "Jugo|Bebida|Agua|Refresco|Gaseosa|Té|Energética|Mate|Pepsi|Cerveza|Avena|Vino|Ron|Whisky|Fernet|Cola|Sprite|Licor|Vodka": "Bebidas",
    "Manzana|Banano|Naranja|Pera|Uva": "Bebidas",
    "Pan|Ponqué|Bizcocho|Panela|Mermelada|Manteca": "Panadería",
    "Yogur|Leche|Queso|Mantequilla": "Lácteos",
    "Arroz|Frijol|Lenteja|Cereal|Frutos secos|Garbanzos|Granola": "Granos y Cereales",
    "Tomate|Cebolla|Papa|Lechuga|Zanahoria|Verduras": "Verduras",
    "Detergente|Jabón|Limpiador|Desinfectante|Lacandina|Shampoo|Servilletas|Cepillo|Mascarilla|Limpiavidrios|Esponjas|Desodorante": "Limpieza",
    "Galleta|Chocolate|Dulce|Confite|Alfajor|Maní|Turrón|Azúcar|Caramelo|Chupetín|Stevia|Pizza|Helado|Aceitunas|Galletitas|Chicle Menta": "Snacks y Dulces"
}

```

#### Buscar productos candidatos para recategorización
```
sugerencias = []

for patron, categoria_sugerida in reglas_categoria.items():
    mask = productos['nombre_producto'].str.contains(patron, case=False, na=False)
    df_sugerido = productos.loc[mask & (productos['categoria'] != categoria_sugerida), 
                                ['id_producto', 'nombre_producto', 'categoria']]
    if not df_sugerido.empty:
        df_sugerido['categoria_sugerida'] = categoria_sugerida
        sugerencias.append(df_sugerido)

```
#### Concatenar resultados
```
if sugerencias:
    sugerencias_df = pd.concat(sugerencias, ignore_index=True)
    print("🔍 Productos potencialmente mal categorizados (según palabra clave):")
    display(sugerencias_df)
else:
    print("✅ No se detectaron productos fuera de su categoría esperada.")

```

#### Aplicar todas las sugerencias automáticamente
Si revisaste sugerencias_df y estás de acuerdo con todas las correcciones sugeridas:

#### Aplicar todas las sugerencias automáticamente
```
for _, fila in sugerencias_df.iterrows():
    productos.loc[productos['id_producto'] == fila['id_producto'], 'categoria'] = fila['categoria_sugerida']

```	
#### Verificar los cambios

Después de hacer tus correcciones:

```
print("Categorías finales actualizadas:")
display(productos['categoria'].value_counts())

```
#### Opcional: vista previa por categoría
```
for cat in productos['categoria'].unique():
    subset = productos[productos['categoria'] == cat].head(20)
    print(f"\n{cat}:")
    display(subset[['id_producto', 'nombre_producto']])

```
**Revisión y recategorización de productos**

| Sección                            | Funcionalidad                                                                     |
| ---------------------------------- | --------------------------------------------------------------------------------- |
| **Vista previa**                   | Tabla de productos ordenable y paginada.                                          |
| **Análisis de categorías**         | Lista única y conteo por categoría.                                               |
| **Normalización automática**       | Aplica reglas de texto con conteo de cambios.                                     |
| **Reglas interactivas**            | Permite editar el diccionario de recategorización directamente desde la interfaz. |
| **Búsqueda automática**            | Encuentra productos con categoría incorrecta según patrones.                      |
| **Tabla ordenable de sugerencias** | Vista paginada con productos a corregir.                                          |
| **Confirmación y guardado**        | Aplica los cambios al dataset en memoria (se reflejarán globalmente).             |
| **Resumen final**                  | Conteo por categoría y ejemplos de productos actualizados.                        |

___
### Análisis estadístico del dataset de ventas

**Distribución de precios**
La variable `precio_unitario` sigue una distribución sesgada a la derecha (asimetría positiva).

**Correlaciones relevantes**
Existe una correlación negativa (-0.65) entre precio y cantidad, lo cual indica...

**Detección de outliers**
Se identificaron 4 productos con precios fuera del rango intercuartílico (IQR).

**Conclusiones**
El análisis muestra que las categorías “Bebidas” y “Snacks y Dulces” concentran la mayor parte del volumen de ventas.

**Qué se encuentra en este módulo?**

| Funcionalidad               | Qué hace                              | Gráfico               |
| --------------------------- | ------------------------------------- | --------------------- |
| **Estadística descriptiva** | Calcula medidas básicas e histograma  | `histplot`            |
| **Medidas de posición**     | Cuartiles, rango y boxplot            | `boxplot`             |
| **Correlaciones**           | Correlaciones y dispersión            | `heatmap` + `scatter` |
| **Confiabilidad**           | Coeficiente de variación (CV)         | `histplot`            |
| **Visualizaciones**         | Boxplot, violinplot, heatmap dinámico | `seaborn`             |


___

# 3. Análisis Estadístico y Visualización de Datos

Fase avanzada de análisis analítico y de negocio, donde buscamos una visión 360° del dataset, integrando ventas, productos, clientes y detalle de ventas.

Lo que estás describiendo es lo que en BI y Data Science llamamos un “Data Mart de Ventas” o “tabla maestra analítica”, la base para análisis estadísticos, dashboards y modelos predictivos.

**🎯 Objetivo**

Crear una tabla analítica unificada (o vista maestra virtual) que consolide todos los datos relevantes en una sola estructura sin perder la integridad entre relaciones, para poder:

* Analizar ventas globales, por producto, cliente, categoría, etc.
* Generar métricas derivadas (importe total, promedio por cliente, ticket medio, etc.).
* Calcular correlaciones entre variables de distintas tablas.
* Usar herramientas de visualización o IA sin necesidad de hacer joins manuales cada vez.

### 🧠 Qué podrás analizar desde la tabla maestra
#### 📊 Análisis cuantitativo

* Importe total por cliente: agrupa c`liente → sum(importe_total)`.
* Ticket promedio por compra: promedio de `importe_total por id_venta`.
* Productos más vendidos: agrupa `nombre_producto → sum(cantidad)`.
* Categorías más rentables: agrupa `categoria → sum(importe_total)`.
* Ciudades con más ventas: agrupa `ciudad → sum(importe_total)`.

#### 📅 Análisis temporal

* Si tienes fecha de venta (fecha_venta), podrás:
* Ventas por mes, trimestre o año.
* Comparativas de crecimiento.
* Detección de estacionalidad (gráficos de línea o barras).

#### 👥 Análisis demográfico (en el futuro)

* Si agregas columnas como `género, edad, ciudad`, podrás:
* Comparar comportamiento por género o edad.
* Ver qué grupos compran más por categoría.
* Identificar clientes frecuentes o nuevos.

#### 🔗 Correlaciones globales

* `cantidad ↔ precio_unitario` → Elasticidad de demanda.
* `importe_total ↔ categoria` → Qué categorías generan más valor.
* `cliente ↔ ciudad` → Concentración geográfica.

#### 📈 Visualizaciones útiles con la tabla maestra
| Análisis                          | Gráfico sugerido                          | Librería         |
| --------------------------------- | ----------------------------------------- | ---------------- |
| Ventas por categoría              | `sns.barplot` o `plotly.express.bar`      | Seaborn / Plotly |
| Productos top                     | `sns.barplot` ordenado por cantidad total | Seaborn          |
| Distribución de importes          | `sns.histplot`                            | Seaborn          |
| Relación cantidad vs precio       | `sns.scatterplot`                         | Seaborn          |
| Heatmap de correlaciones globales | `sns.heatmap`                             | Seaborn          |

#### 💡 Ejemplo de análisis posible desde la "tabla_maestra"
* Una vez seleccionada en el menú:
* Top 10 productos más vendidos: nombre_producto vs cantidad
* Ventas por categoría: categoria vs importe_total
* Clientes con más compras: nombre_cliente vs importe_total
* Correlación cantidad-precio: analiza elasticidad de demanda
* Ventas por ciudad o mes: ciudad vs importe_total, o usando la fecha de venta
* Variabilidad del ticket promedio: usa el coeficiente de variación

___

# 4. Reportes

Fase gerencial / de inteligencia de negocio, dashboard ejecutivo interactivo 📊

**Este nuevo módulo se llama reportes_view.py y estará enfocado en:**
* Mostrar indicadores clave (KPIs) calculados en tiempo real.
* Generar gráficos estratégicos (ventas por producto, cliente, categoría, mes, etc.).
* Ofrecer interpretaciones automáticas para apoyar la toma de decisiones.

#### ✅ Qué incluye este módulo
| Sección                    | Funcionalidad                                                         | Valor para negocio                       |
| -------------------------- | --------------------------------------------------------------------- | ---------------------------------------- |
| **KPIs principales**       | Muestra ventas totales, clientes, ticket promedio, productos vendidos | Da una visión rápida de desempeño        |
| **Top productos**          | Ranking de ventas por cantidad                                        | Identifica productos estrella            |
| **Categorías rentables**   | Ranking por valor total                                               | Detecta líneas de negocio más valiosas   |
| **Top clientes**           | Ranking por cliente                                                   | Identifica compradores clave             |
| **Ventas por mes**         | Línea de tendencia temporal                                           | Revela estacionalidad o picos de demanda |
| **Correlaciones globales** | Mapa de calor                                                         | Muestra qué factores afectan las ventas  |

#### 🧠 Interpretaciones automáticas

Cada gráfico incluye un insight contextual generado automáticamente
(ejemplo: “El producto más vendido es X”, “El mes más fuerte fue julio…”).

| Categoría     | Acción                                                                           | Resultado                                             |
| ------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------- |
| 🗓️ Temporal  | Detecta automáticamente una columna de fecha (`fecha_venta`, `fecha_alta`, etc.) | Crea columnas `año`, `mes`, `mes_texto`, `trimestre`  |
| 🧍 Cliente    | Calcula total de ventas por cliente                                              | Añade `total_cliente` y `% de participación`          |
| 🏷️ Categoría | Calcula total de ventas por categoría                                            | Añade `total_categoria` y `% de participación global` |
| 📦 Producto   | Calcula total de ventas por producto                                             | Añade `total_producto`                                |
| 🧾 Venta      | Calcula ticket promedio por venta                                                | Añade `ticket_venta`                                  |

#### 💡 Ejemplo del resultado

Después de crear la tabla maestra, ahora tendrás automáticamente columnas como:
id_venta | id_cliente | nombre_cliente | nombre_producto | categoria | cantidad | precio_unitario | importe_total | año | mes | mes_texto | trimestre | total_cliente | participacion_cliente_% | total_categoria | participacion_categoria_% | total_producto | ticket_venta



___
# Información del autor del proyecto

**Desarrollado por: Edwar Jaramillo**
**Contacto: [Perfil github](https://github.com/eajaramillo)**
**Contacto: [Proyecto Aurelion](https://github.com/eajaramillo/guayerd_IBM/tree/main/Edwar%20Jaramillo%20-%20Aurelion)**

**Recursos de markdows útiles**
Markdown: [Sintaxis de escritura y formato básicos](https://docs.github.com/es/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
Iconos: [Complete list of github markdown emoji markup](https://gist.github.com/rxaviers/7360908)

```
(Versión README para Github)

Recursos
Pandas Cheat Sheet for Data Science in Python
https://www.datacamp.com/cheat-sheet/pandas-cheat-sheet-for-data-science-in-python
```