"""Etapa 2: IQ -> chirps -> FRFT -> métricas -> candidatos por dispositivo."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from configuracion import (RUTA_RESULTADOS, SEMILLA, ORDEN_FRFT, CHIRPS_POR_CANDIDATO,
                           MODO_AGRUPACION, DESCARTAR_INCOMPLETOS, NORMALIZAR_MAGNITUD,
                           ELIMINAR_ENVOLVENTE)
from fast_frfft import frft_ozaktas

# Se reutilizan las funciones de detección/alineamiento y métricas del script proporcionado.
import procesamiento_base as base


def cargar_preparacion():
    spec = importlib.util.spec_from_file_location('preparar_datos', Path(__file__).with_name('01_preparar_datos.py'))
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def extraer_metricas_archivo(ruta, dispositivo, categoria):
    prep = cargar_preparacion()
    iq = prep.validar_iq(ruta)
    chirps, info = base.preparar_chirps(iq)
    filas = []
    base.NORMALIZAR_MAGNITUD = NORMALIZAR_MAGNITUD
    for indice, chirp in enumerate(chirps):
        if ELIMINAR_ENVOLVENTE:
            chirp = base.eliminar_envolvente(chirp)
        transformada = frft_ozaktas(chirp, ORDEN_FRFT)
        metricas = base.calcular_metricas(transformada)
        filas.append(dict(dispositivo=dispositivo, categoria=categoria,
                          archivo=str(ruta), chirp=indice, **metricas))
    return filas


def generar_metricas(grupos):
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    cache = RUTA_RESULTADOS / 'metricas_por_chirp.csv'
    filas = []
    for dispositivo, (categoria, archivos) in grupos.items():
        print(f'\n{dispositivo} [{categoria}]: {len(archivos)} archivos')
        for j, archivo in enumerate(archivos, 1):
            try:
                filas.extend(extraer_metricas_archivo(archivo, dispositivo, categoria))
            except Exception as exc:
                print(f'  OMITIDO {archivo.name}: {exc}')
            if j % 10 == 0 or j == len(archivos):
                print(f'  {j}/{len(archivos)} archivos')
    if not filas:
        raise RuntimeError('No se pudieron extraer métricas de ningún chirp.')
    df = pd.DataFrame(filas)
    df.to_csv(cache, index=False, encoding='utf-8-sig')
    return df


def construir_candidatos(df_chirps):
    if CHIRPS_POR_CANDIDATO < 1:
        raise ValueError('CHIRPS_POR_CANDIDATO debe ser positivo.')
    if MODO_AGRUPACION not in ('consecutivo', 'aleatorio'):
        raise ValueError('MODO_AGRUPACION debe ser consecutivo o aleatorio.')
    rng = np.random.default_rng(SEMILLA)
    metricas = ['varianza', 'mediana', 'asimetria', 'curtosis', 'factor_cresta', 'entropia']
    candidatos, procedencias = [], []
    for dispositivo, grupo in df_chirps.groupby('dispositivo', sort=True):
        grupo = grupo.sort_values(['archivo', 'chirp']).reset_index(drop=True)
        if MODO_AGRUPACION == 'aleatorio':
            grupo = grupo.iloc[rng.permutation(len(grupo))].reset_index(drop=True)
        n = len(grupo)
        limite = n // CHIRPS_POR_CANDIDATO if DESCARTAR_INCOMPLETOS else int(np.ceil(n / CHIRPS_POR_CANDIDATO))
        for j in range(limite):
            trozo = grupo.iloc[j*CHIRPS_POR_CANDIDATO:(j+1)*CHIRPS_POR_CANDIDATO]
            if trozo.empty:
                continue
            cid = f'{dispositivo.replace("/", "_").replace(chr(92), "_")}_c{j+1:04d}'
            fila = dict(candidato=cid, dispositivo=dispositivo, categoria=trozo['categoria'].iloc[0],
                        n_chirps=len(trozo), n_archivos=trozo['archivo'].nunique())
            fila.update(trozo[metricas].mean().to_dict())
            candidatos.append(fila)
            procedencias.extend(dict(candidato=cid, archivo=r.archivo, chirp=r.chirp)
                                for r in trozo.itertuples(index=False))
        print(f'{dispositivo}: {n} chirps -> {limite} candidatos')
    if not candidatos:
        raise ValueError('No hay suficientes chirps para construir candidatos completos. Reduce CHIRPS_POR_CANDIDATO.')
    resultado = pd.DataFrame(candidatos)
    resultado.to_csv(RUTA_RESULTADOS / 'candidatos.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(procedencias).to_csv(RUTA_RESULTADOS / 'procedencia_candidatos.csv', index=False, encoding='utf-8-sig')
    return resultado
