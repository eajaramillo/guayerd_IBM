# visor_documentacion_md_buscar.py
# -*- coding: utf-8 -*-
"""
Programa de consola sencillo para leer 'documentacion.md' y navegar por secciones.
Ahora incluye:
- Opción 10: Buscar por palabra clave en todas las secciones.
- Opción 11: Exportar una sección a .txt o .md

Requisitos del archivo 'documentacion.md':
- Estar en la misma carpeta que este script.
- Secciones separadas por una línea que contenga exactamente: ___
- Cada sección inicia con un título que empieza con '## <n>.' (n en 1..9)
"""

import os
import re
import sys

DELIMITADOR = "___"
ARCHIVO = "documentacion.md"

MENU = [
    "Seleccione una de las opciones",
    "1. Tema",
    "2. Problema",
    "3. Solución",
    "4. Características set de datos",
    "5. Pasos",
    "6. Pseudocódigo",
    "7. Diagrama de flujo",
    "8. Ejecutar el programa",
    "9. Sugerencias y mejoras aplicadas con Copilot",
    "10. Buscar por palabra clave",
    "11. Exportar una sección (.txt o .md)",
    "0. Para salir",
]

def cargar_secciones(ruta_md):
    """Devuelve un diccionario del tipo {numero (int): texto_seccion (str)}."""
    try:
        with open(ruta_md, "r", encoding="utf-8") as f:
            contenido = f.read()
    except FileNotFoundError:
        print(f"No se encontró '{ruta_md}'. Asegúrate de ejecutarlo en la carpeta correcta.")
        sys.exit(1)

    # Partir por el delimitador de secciones
    partes = [p.strip() for p in contenido.split(DELIMITADOR) if p.strip()]

    secciones = {}
    patron_titulo = re.compile(r"^##\s*(?:[^\d]*)?(\d+)\.\s*(.*)$", re.MULTILINE)

    for bloque in partes:
        m = patron_titulo.search(bloque)
        if not m:
            # No se detectó título numerado, se ignora
            continue
        num = int(m.group(1))
        # Guardamos el bloque completo como sección (tal cual está en el .md)
        secciones[num] = bloque.strip()

    return secciones

def mostrar_menu():
    print("\n" + "="*60)
    for linea in MENU:
        print(linea)
    print("="*60)

def leer_opcion():
    while True:
        op = input("Elige una opción (0-11): ").strip()
        if op.isdigit() and 0 <= int(op) <= 11:
            return int(op)
        print("Opción no válida. Intenta de nuevo.")

def mostrar_seccion(secciones, n):
    texto = secciones.get(n)
    if texto is None:
        print(f"No se encontró la sección ## {n}. en el archivo.")
        return
    print("\n" + "-"*60)
    print(texto)
    print("-"*60 + "\n")

def buscar_palabra(secciones):
    q = input("Ingresa la palabra o frase a buscar: ").strip()
    if not q:
        print("Búsqueda vacía. Volviendo al menú...")
        return
    q_low = q.lower()
    hallazgos = []
    for n, txt in sorted(secciones.items()):
        if q_low in txt.lower():
            # Obtén una vista previa (primera línea del título o primera coincidencia)
            lineas = txt.splitlines()
            titulo = next((l for l in lineas if l.strip().startswith("##")), f"## {n}.")
            # Buscar una línea con coincidencia para mostrar contexto
            contexto = next((l for l in lineas if q_low in l.lower()), "")
            hallazgos.append((n, titulo.strip(), contexto.strip()))
    if not hallazgos:
        print(f'No se encontraron coincidencias para: "{q}".')
        return
    print("\nResultados de búsqueda:")
    for n, titulo, contexto in hallazgos:
        ctx = f" — {contexto}" if contexto else ""
        print(f"- Sección {n}: {titulo}{ctx}")
    # Oferta de abrir alguna sección
    elec = input("¿Deseas ver una sección encontrada? Ingresa su número (o Enter para omitir): ").strip()
    if elec.isdigit():
        mostrar_seccion(secciones, int(elec))

def exportar_seccion(secciones):
    num_str = input("Número de sección a exportar (1-9): ").strip()
    if not (num_str.isdigit() and 1 <= int(num_str) <= 9):
        print("Número no válido.")
        return
    n = int(num_str)
    texto = secciones.get(n)
    if texto is None:
        print(f"No se encontró la sección ## {n}. en el archivo.")
        return

    ext = input("Formato destino (.txt o .md) [md]: ").strip().lower()
    if ext not in (".txt", ".md", "txt", "md", ""):
        print("Formato no válido. Usa .txt o .md.")
        return
    ext = ".md" if ext in ("", "md") else ".txt"

    nombre_def = f"seccion_{n}{ext}"
    nombre = input(f"Nombre de archivo destino [{nombre_def}]: ").strip() or nombre_def
    # El archivo se guarda en la carpeta actual
    try:
        with open(nombre, "w", encoding="utf-8") as f:
            f.write(texto)
        print(f"Sección {n} exportada correctamente a '{nombre}'.")
    except Exception as e:
        print(f"Error exportando archivo: {e}")

def main():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), ARCHIVO)
    secciones = cargar_secciones(ruta)

    print('Bienvenido. Leyendo secciones desde "{}"...'.format(ARCHIVO))

    while True:
        mostrar_menu()
        opcion = leer_opcion()
        if opcion == 0:
            print("Gracias por su atención. ¡Hasta pronto!")
            break
        elif 1 <= opcion <= 9:
            mostrar_seccion(secciones, opcion)
        elif opcion == 10:
            buscar_palabra(secciones)
        elif opcion == 11:
            exportar_seccion(secciones)
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")