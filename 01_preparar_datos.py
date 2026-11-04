"""Etapa 1: lectura/validación y estandarización sin fuga de etiquetas."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from configuracion import RUTA_SENALES, CATEGORIAS, SEMILLA, MAX_ARCHIVOS_POR_DISPOSITIVO


def descubrir_archivos():
    if not RUTA_SENALES.is_dir():
        raise FileNotFoundError(f'No existe la carpeta: {RUTA_SENALES.resolve()}')
    grupos = {}
    for carpeta in sorted(p for p in RUTA_SENALES.rglob('*') if p.is_dir()):
        archivos = sorted(carpeta.glob('*.npy'))
        if not archivos:
            continue
        if MAX_ARCHIVOS_POR_DISPOSITIVO is not None:
            rng = np.random.default_rng(SEMILLA)
            ids = np.sort(rng.choice(len(archivos), min(len(archivos), MAX_ARCHIVOS_POR_DISPOSITIVO), replace=False))
            archivos = [archivos[i] for i in ids]
        # La carpeta identifica un ejemplar/condición, sin mezclar ejemplares.
        dispositivo = str(carpeta.relative_to(RUTA_SENALES))
        grupos[dispositivo] = (CATEGORIAS.get(carpeta.name, carpeta.name), archivos)
    if not grupos:
        raise ValueError('No se encontraron archivos .npy en subcarpetas.')
    return grupos


def validar_iq(ruta, muestras_esperadas=1_000_000):
    x = np.load(ruta, mmap_mode='r', allow_pickle=False)
    if not np.iscomplexobj(x):
        raise ValueError('Se esperaba IQ complejo.')
    if x.size != muestras_esperadas:
        raise ValueError(f'Longitud {x.size}; esperada {muestras_esperadas} (50 ms, 20 MHz).')
    if not np.isfinite(x).all():
        raise ValueError('Contiene NaN o infinito.')
    return x.reshape(-1)


def preparar_matriz(df, columnas):
    if len(df) < 3:
        raise ValueError('Se necesitan al menos 3 candidatos.')
    if df[columnas].isna().any().any() or not np.isfinite(df[columnas].to_numpy(dtype=float)).all():
        raise ValueError('Hay características no finitas.')
    X = df[columnas].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    constantes = [c for c, v in zip(columnas, scaler.var_) if v < 1e-20]
    if constantes:
        print('ADVERTENCIA: columnas prácticamente constantes:', constantes)
    return Xz, scaler
