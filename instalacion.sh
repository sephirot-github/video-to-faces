#!/bin/bash

# Este script crea un entorno virtual y instala todas las dependencias del proyecto.

# 1. Crea el entorno virtual
echo "Creando el entorno virtual..."
python3 -m venv venv

# 2. Activa el entorno virtual. Si falla, el script se detiene.
echo "Activando el entorno virtual..."
source venv/bin/activate || { echo "Error: No se pudo activar el entorno virtual. La instalación fallará."; exit 1; }

# 3. Instala el proyecto en modo editable (usando setup.py)
echo "Instalando las dependencias del proyecto..."
pip install -e .

# 4. Desactiva el entorno virtual
deactivate

echo "¡Instalación completa! Para ejecutar la aplicación, usa el script iniciar.sh"