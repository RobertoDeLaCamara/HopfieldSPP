# src/__init__.py
# Este archivo puede estar vacío si no necesitas lógica aquí.
__version__ = "1.0.0"
__author__ = "Roberto de la Cámara (roberto.de.la.camara.garcia@gmail.com)"

import logging
# Configuración global de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)