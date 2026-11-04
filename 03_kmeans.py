"""Etapa 3: K-means con selección exploratoria de K por silueta."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from configuracion import (EXPERIMENTOS, K_MIN, K_MAX, SEMILLA,
                           N_INICIALIZACIONES, RUTA_RESULTADOS)


def _preparar():
    spec = importlib.util.spec_from_file_location('preparar_datos', Path(__file__).with_name('01_preparar_datos.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ejecutar_experimentos(candidatos):
    preparar_matriz = _preparar().preparar_matriz
    resultados = {}
    for nombre, columnas in EXPERIMENTOS.items():
        X, escalador = preparar_matriz(candidatos, columnas)
        n = len(X)
        limite = min(K_MAX, n - 1)
        k_valores = range(max(2, K_MIN), limite + 1)
        registros, modelos = [], {}
        for k in k_valores:
            modelo = KMeans(n_clusters=k, n_init=N_INICIALIZACIONES, random_state=SEMILLA)
            etiquetas = modelo.fit_predict(X)
            if len(np.unique(etiquetas)) < 2:
                continue
            silueta = silhouette_score(X, etiquetas)
            registros.append({'experimento': nombre, 'k': k, 'silueta': silueta,
                              'inercia': modelo.inertia_})
            modelos[k] = modelo
        if not registros:
            print(f'Experimento {nombre}: no se pudo evaluar K; omitido.')
            continue
        tabla = pd.DataFrame(registros)
        mejor_k = int(tabla.loc[tabla['silueta'].idxmax(), 'k'])
        modelo = modelos[mejor_k]
        asignaciones = candidatos.copy()
        asignaciones['cluster'] = modelo.labels_ + 1
        asignaciones['experimento'] = nombre
        asignaciones['k_elegido'] = mejor_k
        tabla.to_csv(RUTA_RESULTADOS / f'experimento_{nombre}_seleccion_k.csv', index=False, encoding='utf-8-sig')
        asignaciones.to_csv(RUTA_RESULTADOS / f'experimento_{nombre}_asignaciones.csv', index=False, encoding='utf-8-sig')
        resultados[nombre] = (asignaciones, tabla, X, modelo)
        print(f'Experimento {nombre}: K={mejor_k}, silueta={tabla.silueta.max():.3f}')
    return resultados
