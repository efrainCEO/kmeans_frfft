# -*- coding: utf-8 -*-
"""Extracción de métricas FRFT (orden fijo, con varianza) sobre ráfagas IQ.
Requiere fast_frfft.py en el mismo directorio.
Dependencias: numpy, scipy, matplotlib.
"""
from pathlib import Path
import random
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from fast_frfft import frft_ozaktas

# ==================== CONFIGURACIÓN ====================
PATH_BASE = Path('./señales')
# None: detectar todas las subcarpetas que contengan .npy.
# También puedes especificar ['T_Cam_Down_1', 'T_Cam_Down_2', ...].
FOLDERS = None
NUM_FILES = 3              # None para procesar todos los archivos por carpeta
NUM_CHIRPS = 5             # None para todos los chirps disponibles por archivo
SEMILLA = 123
ORDEN_FRFT = 0.3
FS = 20_000_000
T_CHIRP = 5e-3
T_SILENCIO = 10e-3
N_CHIRP = int(round(T_CHIRP * FS))
N_SILENCIO = int(round(T_SILENCIO * FS))
N_CICLO = N_CHIRP + N_SILENCIO
N_BLOQUES_DETECCION = 1000
MAX_SHIFT = 1000
MIN_CICLOS = 1
ELIMINAR_ENVOLVENTE = False
UMBRAL_RELATIVO = 0.05
NORMALIZAR_MAGNITUD = False  # True: divide |FRFT| entre su RMS antes de métricas
CARPETA_SALIDA = Path('resultados_metricas_frft_03')
MOSTRAR_GRAFICAS = True
GUARDAR_FIGURAS = True

# Asigna categorías para agrupar carpetas distintas de una misma clase.
# Las carpetas no incluidas se conservan como categoría independiente.
CATEGORIAS = {
    'T_Audi_Down_1': 'Audífonos', 
    'T_Audi_Down_2': 'Audífonos',
    'T_Cam_Down_1': 'Cámaras', 
    'T_Cam_Down_2': 'Cámaras',
    'T_Metal_Up_1': 'Metal',
    'T_SiTX_Down_1': 'Solo TX', 
    'T_NoTX': 'Solo ruido',
    # 'T_Cel_Down_1': 'Celulares', 
    # 'T_Cel_Down_2': 'Celulares',
}

# ==================== DETECCIÓN Y ALINEAMIENTO ====================


def calcular_envolvente_reducida(
    senal,
    n_bloques
):
    """
    Calcula una envolvente de magnitud reducida.

    Se utiliza exclusivamente para detección temporal.
    """

    magnitud = np.abs(senal)

    n_bloques = min(
        n_bloques,
        magnitud.size
    )

    bloques = np.array_split(
        magnitud,
        n_bloques
    )

    medias = np.array(
        [np.mean(b) for b in bloques],
        dtype=np.float64
    )

    return medias

def detectar_inicio_chirp(senal):
    """
    Estima la posición del inicio de un chirp dentro
    de un periodo de 15 ms.

    Método:
    1. Reducir la envolvente de magnitud.
    2. Probar posiciones dentro de un periodo.
    3. Comparar energía de la ventana activa de 5 ms
       contra la ventana de silencio de 10 ms.
    4. Elegir la posición con mayor contraste.

    Nota:
    La detección es aproximada y depende de que la
    respuesta activa sea distinguible del silencio.
    """

    envolvente = calcular_envolvente_reducida(
        senal,
        N_BLOQUES_DETECCION
    )

    muestras_por_bloque = (
        len(senal) / len(envolvente)
    )

    bloques_chirp = max(
        1,
        int(round(
            N_CHIRP / muestras_por_bloque
        ))
    )

    bloques_ciclo = max(
        1,
        int(round(
            N_CICLO / muestras_por_bloque
        ))
    )

    bloques_silencio = max(
        1,
        bloques_ciclo - bloques_chirp
    )

    scores = np.full(
        bloques_ciclo,
        -np.inf,
        dtype=np.float64
    )

    # Evaluar candidatos en todo un periodo.
    for fase in range(bloques_ciclo):

        contrastes = []

        # Permite fases negativas: una ventana periódica
        # puede comenzar antes del inicio del archivo.
        k_min = int(
            np.floor(
                -fase / bloques_ciclo
            )
        )

        k_max = int(
            np.ceil(
                len(envolvente) / bloques_ciclo
            )
        )

        for k in range(k_min, k_max + 1):

            pos = fase + k * bloques_ciclo

            # Exigir chirp y silencio completos.
            if pos < 0:
                continue

            fin_chirp = pos + bloques_chirp
            fin_ciclo = pos + bloques_ciclo

            if fin_ciclo > len(envolvente):
                continue

            activo = envolvente[
                pos:fin_chirp
            ]

            silencio = envolvente[
                fin_chirp:fin_ciclo
            ]

            if (
                len(activo) == 0
                or len(silencio) == 0
            ):
                continue

            contraste = (
                np.mean(activo)
                - np.mean(silencio)
            )

            contrastes.append(contraste)

        if contrastes:
            scores[fase] = np.mean(
                contrastes
            )

    if not np.any(np.isfinite(scores)):
        raise ValueError(
            "No se pudo estimar el inicio del chirp."
        )

    fase_optima = int(
        np.argmax(scores)
    )

    inicio = int(
        round(
            fase_optima
            * muestras_por_bloque
        )
    )

    inicio %= N_CICLO

    return inicio, scores

def extraer_ciclos(
    senal,
    inicio
):
    """
    Extrae ciclos completos de N_CICLO muestras.

    No se utilizan fragmentos incompletos.
    """

    senal = np.asarray(
        senal,
        dtype=np.complex128
    ).reshape(-1)

    ciclos = []

    posicion = int(inicio)

    while (
        posicion + N_CICLO
        <= len(senal)
    ):

        ciclo = senal[
            posicion:
            posicion + N_CICLO
        ]

        ciclos.append(ciclo)

        posicion += N_CICLO

    if len(ciclos) < MIN_CICLOS:
        raise ValueError(
            "No hay suficientes ciclos completos."
        )

    return np.asarray(ciclos)

def desplazar_sin_wrap(
    x,
    shift
):
    """
    Desplaza una señal sin envolver muestras.

    A diferencia de np.roll, las muestras desplazadas
    fuera del intervalo no reaparecen en el extremo opuesto.

    Los huecos se rellenan con cero.
    """

    salida = np.zeros_like(x)

    if shift == 0:
        return x.copy()

    if abs(shift) >= len(x):
        return salida

    if shift > 0:
        salida[shift:] = x[:-shift]

    else:
        salida[:shift] = x[-shift:]

    return salida

def alinear_ciclos(
    ciclos,
    max_shift=MAX_SHIFT
):
    """
    Alinea ciclos utilizando correlación de magnitudes.

    La referencia es el primer ciclo.

    La correlación se calcula sobre una región cercana
    al chirp activo.

    Se aplica el desplazamiento al IQ complejo.

    No se realiza promedio temporal.
    """

    ciclos = np.asarray(
        ciclos,
        dtype=np.complex128
    )

    if len(ciclos) == 1:
        return ciclos.copy(), np.array([0])

    # Incluir una pequeña región adicional.
    n_region = min(
        N_CICLO,
        N_CHIRP + max_shift
    )

    referencia = np.abs(
        ciclos[0, :n_region]
    ).astype(np.float64)

    referencia -= referencia.mean()

    ciclos_alineados = [
        ciclos[0].copy()
    ]

    shifts = [0]

    for i in range(1, len(ciclos)):

        region = np.abs(
            ciclos[i, :n_region]
        ).astype(np.float64)

        region -= region.mean()

        correlacion = correlate(
            referencia,
            region,
            mode="full",
            method="fft"
        )

        lags = correlation_lags(
            len(referencia),
            len(region),
            mode="full"
        )

        mascara = (
            np.abs(lags) <= max_shift
        )

        corr_local = correlacion[mascara]
        lags_local = lags[mascara]

        mejor_shift = int(
            lags_local[
                np.argmax(corr_local)
            ]
        )

        ciclo_alineado = desplazar_sin_wrap(
            ciclos[i],
            mejor_shift
        )

        ciclos_alineados.append(
            ciclo_alineado
        )

        shifts.append(
            mejor_shift
        )

    return (
        np.asarray(ciclos_alineados),
        np.asarray(shifts)
    )

def eliminar_envolvente(
    senal,
    umbral_relativo=UMBRAL_RELATIVO
):
    """
    Normalización de magnitud:

        y[n] = x[n] / |x[n]|

    Se aplica únicamente donde la amplitud supera
    un umbral relativo.

    Conserva principalmente la información de fase.
    """

    senal = np.asarray(
        senal,
        dtype=np.complex128
    )

    magnitud = np.abs(senal)

    maximo = np.max(magnitud)

    if maximo <= 0:
        return np.zeros_like(senal)

    umbral = (
        umbral_relativo * maximo
    )

    mascara = magnitud > umbral

    salida = np.zeros_like(senal)

    salida[mascara] = (
        senal[mascara]
        / magnitud[mascara]
    )

    return salida

def preparar_chirps(senal):
    """
    Ejecuta una sola vez:

        detectar -> extraer -> alinear -> cortar 5 ms

    Devuelve los chirps complejos listos para FRFT.
    """

    inicio, scores = detectar_inicio_chirp(
        senal
    )

    ciclos = extraer_ciclos(
        senal,
        inicio
    )

    ciclos_alineados, shifts = alinear_ciclos(
        ciclos,
        max_shift=MAX_SHIFT
    )

    chirps = ciclos_alineados[
        :,
        :N_CHIRP
    ]

    return chirps, {
        "inicio": inicio,
        "shifts": shifts,
        "n_ciclos": len(chirps),
        "scores": scores
    }


# ==================== SELECCIÓN Y MÉTRICAS ====================
METRICAS = ('varianza', 'asimetria', 'curtosis', 'mediana', 'entropia', 'factor_cresta')


def listar_carpetas():
    if FOLDERS is not None:
        return [PATH_BASE / f for f in FOLDERS]
    return sorted(p for p in PATH_BASE.rglob('*') if p.is_dir() and any(p.glob('*.npy')))


def seleccionar_archivos(carpeta, rng):
    archivos = sorted(carpeta.glob('*.npy'))
    if NUM_FILES is None:
        return archivos
    return rng.sample(archivos, min(NUM_FILES, len(archivos)))


def seleccionar_chirps(chirps, rng):
    if NUM_CHIRPS is None or NUM_CHIRPS >= len(chirps):
        return chirps
    indices = sorted(rng.sample(range(len(chirps)), NUM_CHIRPS))
    return chirps[indices]


def calcular_metricas(transformada):
    x = np.abs(transformada).astype(np.float64)
    if NORMALIZAR_MAGNITUD:
        rms = np.sqrt(np.mean(x*x))
        if rms > 0:
            x /= rms
    media = np.mean(x)
    centrada = x - media
    m2 = np.mean(centrada**2)
    escala = np.sqrt(m2)
    if m2 > 1e-24:
        z = centrada / escala
        asimetria = float(np.mean(z**3))
        curtosis = float(np.mean(z**4) - 3.0)
    else:
        asimetria, curtosis = 0.0, 0.0
    potencia = x*x
    energia = np.sum(potencia)
    if energia > 0 and len(x) > 1:
        p = potencia / energia
        p = p[p > 0]
        entropia = float(-np.sum(p*np.log2(p))/np.log2(len(x)))
        cresta = float(np.max(x)/np.sqrt(np.mean(potencia)))
    else:
        entropia, cresta = 0.0, 0.0
    return dict(varianza=float(m2), asimetria=asimetria,
                curtosis=curtosis, mediana=float(np.median(x)),
                entropia=entropia, factor_cresta=cresta)


def procesar_archivo(archivo, rng):
    senal = np.load(archivo, mmap_mode='r').reshape(-1)
    if not np.iscomplexobj(senal):
        raise ValueError('El archivo no contiene IQ complejo')
    chirps, info = preparar_chirps(senal)
    chirps = seleccionar_chirps(chirps, rng)
    filas = []
    for indice, chirp in enumerate(chirps):
        if ELIMINAR_ENVOLVENTE:
            chirp = eliminar_envolvente(chirp, UMBRAL_RELATIVO)
        frft = frft_ozaktas(chirp, ORDEN_FRFT)
        filas.append({'chirp': indice, **calcular_metricas(frft)})
    promedio = {m: float(np.mean([r[m] for r in filas])) for m in METRICAS}
    return filas, promedio, info


def guardar_csv(ruta, filas, columnas):
    with ruta.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filas)


def resumir_por_categoria(filas_archivo):
    categorias = sorted({r['categoria'] for r in filas_archivo})
    resumen = []
    for categoria in categorias:
        grupo = [r for r in filas_archivo if r['categoria'] == categoria]
        fila = {'categoria': categoria, 'n_archivos': len(grupo)}
        for m in METRICAS:
            valores = np.array([r[m] for r in grupo])
            fila[m+'_media'] = float(np.mean(valores))
            fila[m+'_std_archivos'] = float(np.std(valores, ddof=1)) if len(valores)>1 else 0.0
        resumen.append(fila)
    return resumen


def graficar_metricas(resumen):
    categorias = [r['categoria'] for r in resumen]
    for metrica in METRICAS:
        medias = [r[metrica+'_media'] for r in resumen]
        dispersion = [r[metrica+'_std_archivos'] for r in resumen]
        fig, ax = plt.subplots(figsize=(max(9, len(categorias)*1.2), 5))
        pos = np.arange(len(categorias))
        ax.bar(pos, medias, yerr=dispersion, capsize=4, alpha=0.8)
        ax.set_xticks(pos, categorias, rotation=25, ha='right')
        ax.set_ylabel(metrica.replace('_', ' ').capitalize())
        ax.set_title(f'{metrica.replace("_", " ").capitalize()} | FRFT a={ORDEN_FRFT}')
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        if GUARDAR_FIGURAS:
            fig.savefig(CARPETA_SALIDA / f'{metrica}_a_{ORDEN_FRFT}.png', dpi=180)
        if MOSTRAR_GRAFICAS:
            plt.show()
        plt.close(fig)


def main():
    t0 = time.perf_counter()
    rng = random.Random(SEMILLA)
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    filas_chirp, filas_archivo = [], []
    carpetas = listar_carpetas()
    if not carpetas:
        raise FileNotFoundError(f'No hay carpetas con .npy en {PATH_BASE.resolve()}')
    for carpeta in carpetas:
        nombre = carpeta.name
        categoria = CATEGORIAS.get(nombre, nombre)
        archivos = seleccionar_archivos(carpeta, rng)
        print(f'\n{nombre} ({categoria}): {len(archivos)} archivos')
        for archivo in archivos:
            try:
                individuales, promedio, info = procesar_archivo(archivo, rng)
                identificacion = {'categoria': categoria, 'carpeta': nombre, 'archivo': str(archivo)}
                filas_chirp.extend([{**identificacion, **r} for r in individuales])
                filas_archivo.append({**identificacion, 'n_chirps': len(individuales), **promedio})
                print(f'  OK {archivo.name}: {len(individuales)} chirps (de {info["n_ciclos"]})')
            except Exception as e:
                print(f'  ERROR {archivo.name}: {e}')
    if not filas_archivo:
        raise RuntimeError('No se procesó ningún archivo correctamente')
    resumen = resumir_por_categoria(filas_archivo)
    guardar_csv(CARPETA_SALIDA/'metricas_chirp.csv', filas_chirp,
                ['categoria','carpeta','archivo','chirp',*METRICAS])
    guardar_csv(CARPETA_SALIDA/'metricas_archivo.csv', filas_archivo,
                ['categoria','carpeta','archivo','n_chirps',*METRICAS])
    columnas = ['categoria','n_archivos'] + [x for m in METRICAS for x in (m+'_media',m+'_std_archivos')]
    guardar_csv(CARPETA_SALIDA/'resumen_categorias.csv', resumen, columnas)
    graficar_metricas(resumen)
    print(f'\nFinalizado: {len(filas_archivo)} archivos, {len(filas_chirp)} chirps; '
          f'{(time.perf_counter()-t0)/60:.2f} minutos')
    print(f'Resultados: {CARPETA_SALIDA.resolve()}')
    return resumen


if __name__ == '__main__':
    resultados = main()
