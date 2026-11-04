# Experimentos A, B y C con K-means sobre FRFT a=0.3

## Instalación

`python -m pip install numpy scipy pandas matplotlib scikit-learn`

## Estructura

Coloca los scripts y una carpeta `señales` (o modifica `RUTA_SENALES` en `configuracion.py`):

```
experimentos_kmeans_frft/
  ejecutar.py
  configuracion.py
  01_preparar_datos.py
  02_crear_candidatos.py
  03_kmeans.py
  04_evaluar.py
  procesamiento_base.py
  fast_frfft.py
  señales/
    T_Cam_Down_1/
      grabacion_001.npy
      grabacion_002.npy
    T_Cam_Down_2/
      grabacion_001.npy
    T_Audi_Down_1/
      grabacion_001.npy
```

`procesamiento_base.py` reutiliza la detección, alineamiento y métricas del script proporcionado. `fast_frfft.py` es la implementación de Özaktas proporcionada. No es necesario ejecutar esos módulos directamente.

## Uso

Desde esta carpeta, ejecutar `python ejecutar.py` o abrir `ejecutar.py` en Spyder con directorio de trabajo configurado a esta carpeta.

Por defecto se procesan todos los archivos `.npy` complejos con 1 000 000 de muestras (50 ms a 20 MHz), se extraen chirps de 5 ms de ciclos de 15 ms, se calcula FRFT a=0.3 y se forman candidatos con 30 chirps cada uno, sin reutilizarlos. Cada carpeta se considera un dispositivo/condición diferente. No se mezclan dispositivos dentro de un candidato.

Los archivos se ordenan **lexicográficamente por nombre**. El modo `consecutivo` aproxima ventanas temporales solamente si esos nombres reflejan el orden real de captura. Si no es así, corrige el orden de nombres o emplea `aleatorio` para una evaluación exploratoria.

Los últimos chirps insuficientes para un candidato se descartan por defecto. Ajusta `CHIRPS_POR_CANDIDATO` para obtener más candidatos. Para una prueba inicial rápida configura `MAX_ARCHIVOS_POR_DISPOSITIVO=10` y `CHIRPS_POR_CANDIDATO=5`.

## Resultados

- `metricas_por_chirp.csv`: métricas individuales por chirp, con archivo y dispositivo de origen.
- `candidatos.csv`: promedio de las métricas por candidato.
- `procedencia_candidatos.csv`: trazabilidad exacta de chirps utilizados.
- `experimento_A/B/C_asignaciones.csv`: candidato, dispositivo, categoría y clúster asignado.
- `experimento_A/B/C_seleccion_k.csv`: silueta e inercia para cada K.
- `experimento_A/B/C/contingencia_dispositivo.csv` y `contingencia_categoria.csv`.
- `experimento_A/B/C/pca.png`, `silueta_por_k.png`.
- `comparacion_ABC.csv`: comparación de silueta, ARI y NMI.

## Advertencias metodológicas

- La estandarización se hace **después** de calcular los promedios y **por experimento**, sin utilizar etiquetas.
- El valor K se elige por silueta dentro de 2..9 (o hasta N-1); puede ser diferente entre A/B/C.
- ARI y NMI comparan clústeres con etiquetas conocidas **solo para evaluación**.
- El promedio de 30 chirps no equivale automáticamente a una grabación continua de 450 ms si los archivos no son contiguos.
- Los candidatos provenientes de una misma sesión no son experimentos independientes: no interpretar los índices como validación de generalización.
- Las condiciones `Solo TX` y `Solo ruido` son referencias, no necesariamente respuestas de dispositivos.
- No se aplica un umbral SNR automático: este análisis es exploratorio.
