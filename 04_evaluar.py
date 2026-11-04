"""Etapa 4: tablas cluster-dispositivo, métricas y visualizaciones PCA."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from configuracion import RUTA_RESULTADOS, MOSTRAR_GRAFICAS


def evaluar(resultados):
    resumen = []
    for nombre, (asignaciones, tabla, X, modelo) in resultados.items():
        carpeta = RUTA_RESULTADOS / f'experimento_{nombre}'
        carpeta.mkdir(exist_ok=True, parents=True)
        for campo in ('categoria', 'dispositivo'):
            matriz = pd.crosstab(asignaciones[campo], asignaciones['cluster'])
            matriz.to_csv(carpeta / f'contingencia_{campo}.csv', encoding='utf-8-sig')
            print(f'\nExperimento {nombre} - cluster vs {campo}:\n{matriz.to_string()}')
        etiquetas = asignaciones['cluster'].to_numpy()
        categoria = asignaciones['categoria'].to_numpy()
        dispositivo = asignaciones['dispositivo'].to_numpy()
        resumen.append(dict(experimento=nombre, k=modelo.n_clusters,
                            silueta=float(tabla['silueta'].max()),
                            ARI_categoria=adjusted_rand_score(categoria, etiquetas),
                            NMI_categoria=normalized_mutual_info_score(categoria, etiquetas),
                            ARI_dispositivo=adjusted_rand_score(dispositivo, etiquetas),
                            n_candidatos=len(asignaciones)))
        # PCA se usa solo para visualización; el K-means se ajustó en todas las dimensiones.
        xy = PCA(n_components=2).fit_transform(X)
        fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, campo in zip(ejes, ('cluster', 'categoria')):
            for valor in sorted(asignaciones[campo].unique(), key=str):
                mascara = (asignaciones[campo] == valor).to_numpy()
                ax.scatter(xy[mascara, 0], xy[mascara, 1], s=26, alpha=0.75, label=str(valor))
            ax.set_title(f'Experimento {nombre}: {campo}')
            ax.set_xlabel('Componente principal 1')
            ax.set_ylabel('Componente principal 2')
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(carpeta / 'pca.png', dpi=180)
        if MOSTRAR_GRAFICAS:
            plt.show()
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(tabla['k'], tabla['silueta'], marker='o')
        ax.set(xlabel='Número de grupos K', ylabel='Coeficiente de silueta',
               title=f'Experimento {nombre}: selección de K')
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(carpeta / 'silueta_por_k.png', dpi=180)
        if MOSTRAR_GRAFICAS:
            plt.show()
        plt.close(fig)
    if resumen:
        pd.DataFrame(resumen).to_csv(RUTA_RESULTADOS / 'comparacion_ABC.csv', index=False, encoding='utf-8-sig')
        print('\nRESUMEN A/B/C:\n', pd.DataFrame(resumen).to_string(index=False))
