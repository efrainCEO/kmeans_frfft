"""Ejecutar: python ejecutar.py  (desde la carpeta de este proyecto)."""
import importlib.util
from pathlib import Path
from configuracion import RUTA_RESULTADOS


def cargar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, Path(__file__).with_name(archivo))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def main():
    p = cargar('preparacion', '01_preparar_datos.py')
    c = cargar('candidatos', '02_crear_candidatos.py')
    k = cargar('agrupamiento', '03_kmeans.py')
    e = cargar('evaluacion', '04_evaluar.py')
    grupos = p.descubrir_archivos()
    chirps = c.generar_metricas(grupos)
    candidatos = c.construir_candidatos(chirps)
    resultados = k.ejecutar_experimentos(candidatos)
    e.evaluar(resultados)
    print(f'\nArchivos de salida: {RUTA_RESULTADOS.resolve()}')


if __name__ == '__main__':
    main()
