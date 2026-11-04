# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:38:10 2026

@author: Efrain Santos Luna

@mail: efrain.santos.consultor@gmail.com

@phone:55-66-18-62-95
"""

import numpy as np
from scipy.signal import fftconvolve

# ============================================================
# FRFT DE ÖZAKTAS
# ============================================================

def _interpolacion_sinc(senal):

    n = senal.size

    intercalada = np.zeros(
        2 * n - 1,
        dtype=np.complex128
    )

    intercalada[::2] = senal


    indices = np.arange(
        -(2 * n - 3),
        2 * n - 2
    )


    interpolada = fftconvolve(

        intercalada,

        np.sinc(
            indices / 2
        ),

        mode="full"
    )


    return interpolada[
        2 * n - 3:
        -(2 * n - 3)
    ]


def frft_ozaktas(
    senal,
    orden
):
    """
    Transformada Fraccional de Fourier mediante el algoritmo
    rápido de Özaktas.

    La entrada puede ser una señal compleja.
    """

    x = np.asarray(
        senal,
        dtype=np.complex128
    ).reshape(-1)


    n = x.size


    if n < 2:

        raise ValueError(
            "La FRFT necesita al menos dos muestras."
        )


    orden = float(
        orden
    ) % 4.0


    desplazamiento = (
        np.arange(n)
        + n // 2
    ) % n


    escala = np.sqrt(n)


    # --------------------------------------------------------
    # Casos especiales
    # --------------------------------------------------------

    if np.isclose(
        orden,
        0.0
    ):

        return x.copy()


    if np.isclose(
        orden,
        1.0
    ):

        return (
            np.fft.fft(
                x[desplazamiento]
            )
            / escala
        )


    if np.isclose(
        orden,
        2.0
    ):

        return np.flip(x)


    if np.isclose(
        orden,
        3.0
    ):

        return (
            np.fft.ifft(
                x[desplazamiento]
            )
            * escala
        )


    # --------------------------------------------------------
    # Reducción del orden
    # --------------------------------------------------------

    if orden > 2.0:

        orden -= 2.0

        x = np.flip(x)


    if orden > 1.5:

        orden -= 1.0

        x = (
            np.fft.fft(
                x[desplazamiento]
            )
            / escala
        )


    if orden < 0.5:

        orden += 1.0

        x = (
            np.fft.ifft(
                x[desplazamiento]
            )
            * escala
        )


    angulo = (
        orden
        * np.pi
        / 2.0
    )


    # --------------------------------------------------------
    # Interpolación
    # --------------------------------------------------------

    x = np.concatenate(
        (

            np.zeros(
                n - 1,
                dtype=np.complex128
            ),

            _interpolacion_sinc(
                x
            ),

            np.zeros(
                n - 1,
                dtype=np.complex128
            ),
        )
    )


    indices = np.arange(
        -2 * n + 2,
        2 * n - 1
    )


    chirp_premodulacion = np.exp(

        -1j
        * np.pi
        / n
        * np.tan(
            angulo / 2.0
        )
        * indices**2
        / 4.0
    )


    x *= chirp_premodulacion


    constante = (
        np.pi
        /
        (
            4.0
            * n
            * np.sin(
                angulo
            )
        )
    )


    indices_convolucion = np.arange(
        -(4 * n - 4),
        4 * n - 3
    )


    nucleo = np.exp(

        1j
        * constante
        * indices_convolucion**2
    )


    x = fftconvolve(

        nucleo,

        x,

        mode="full"
    )


    x = (

        x[
            4 * n - 4:
            8 * n - 7
        ]

        * np.sqrt(
            constante
            / np.pi
        )
    )


    x *= chirp_premodulacion


    resultado = x[
        n - 1:
        -(n - 1):
        2
    ]


    resultado *= np.exp(

        -1j
        * (1.0 - orden)
        * np.pi
        / 4.0
    )


    return resultado