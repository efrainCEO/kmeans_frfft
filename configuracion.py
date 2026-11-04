from pathlib import Path

RUTA_SENALES = Path('./señales')
RUTA_RESULTADOS = Path('./resultados_kmeans')
SEMILLA = 123
ORDEN_FRFT = 0.3
# Se procesan TODOS los archivos por defecto; para una prueba rápida poner 10.
MAX_ARCHIVOS_POR_DISPOSITIVO = None
# Cantidad de chirps para formar un candidato, sin reutilizar chirps.
CHIRPS_POR_CANDIDATO = 30
# 'consecutivo': orden natural de nombres de archivo, sin mezclar dispositivos.
# 'aleatorio': permuta los chirps de cada dispositivo antes de agrupar.
MODO_AGRUPACION = 'consecutivo'
# Evita candidatos incompletos: los chirps restantes se descartan.
DESCARTAR_INCOMPLETOS = True
NORMALIZAR_MAGNITUD = False
ELIMINAR_ENVOLVENTE = False
# Si hay carpetas nuevas, quedan como categorías independientes.
CATEGORIAS = {
    'T_Audi_Down_1': 'Audífonos', 
    'T_Audi_Down_2': 'Audífonos',
    'T_Cam_Down_1': 'Cámaras', 
    'T_Cam_Down_2': 'Cámaras',
    'T_Metal_Down_1': 'Metal',
    'T_SiTX_Down_1': 'Solo TX', 
    'T_NoTX': 'Solo ruido',
    'T_NoTX_2': 'Solo ruido',
}
EXPERIMENTOS = {
    'A': ['varianza', 'mediana'],
    'B': ['asimetria', 'curtosis', 'factor_cresta'],
    'C': ['varianza', 'mediana', 'asimetria', 'curtosis', 'factor_cresta'],
}
K_MIN = 4
K_MAX = 4
N_INICIALIZACIONES = 20
MOSTRAR_GRAFICAS = False
