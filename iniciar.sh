#!/bin/bash

# Este script activa el entorno virtual y ejecuta la aplicación de Gradio.

# 1. Activa el entorno virtual
echo "Activando el entorno virtual..."
source venv/bin/activate

# 2. Ejecuta la aplicación de Gradio
echo "Lanzando la aplicación de Gradio..."
python3 app.py

# El entorno se mantendrá activo hasta que cierres la terminal o uses 'deactivate'