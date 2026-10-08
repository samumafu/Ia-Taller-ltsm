"""Arquitecturas LSTM para regresion de demanda energetica.

Este modulo solo define y compila modelos; no carga datos ni realiza
entrenamiento.
"""

from __future__ import annotations

from collections.abc import Callable

from tensorflow import keras


InputShape = tuple[int, int]


def _validar_input_shape(input_shape: InputShape) -> None:
    """Valida la forma ``(ventana, n_features)`` antes de crear un modelo."""

    if len(input_shape) != 2:
        raise ValueError("input_shape debe tener la forma (ventana, n_features).")
    if any(not isinstance(dimension, int) or dimension <= 0 for dimension in input_shape):
        raise ValueError("ventana y n_features deben ser enteros positivos.")


def _compilar(modelo: keras.Model) -> keras.Model:
    """Aplica la configuracion comun de regresion."""

    modelo.compile(optimizer="adam", loss="mse")
    return modelo


def crear_lstm_base(input_shape: InputShape) -> keras.Model:
    """Crea LSTM(64) -> Dense(1)."""

    _validar_input_shape(input_shape)
    modelo = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.LSTM(64),
            keras.layers.Dense(1),
        ],
        name="lstm_base",
    )
    return _compilar(modelo)


def crear_lstm_profunda(input_shape: InputShape) -> keras.Model:
    """Crea la arquitectura LSTM profunda solicitada."""

    _validar_input_shape(input_shape)
    modelo = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1),
        ],
        name="lstm_profunda",
    )
    return _compilar(modelo)


def crear_lstm_propuesta(input_shape: InputShape) -> keras.Model:
    """Crea la arquitectura LSTM propuesta solicitada."""

    _validar_input_shape(input_shape)
    modelo = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1),
        ],
        name="lstm_propuesta",
    )
    return _compilar(modelo)


def crear_modelo(nombre: str, input_shape: InputShape) -> keras.Model:
    """Crea ``base``, ``profunda`` o ``propuesta`` según ``nombre``."""

    constructores: dict[str, Callable[[InputShape], keras.Model]] = {
        "base": crear_lstm_base,
        "profunda": crear_lstm_profunda,
        "propuesta": crear_lstm_propuesta,
    }
    nombre_normalizado = nombre.strip().lower()
    if nombre_normalizado not in constructores:
        opciones = ", ".join(constructores)
        raise ValueError(f"Modelo desconocido: {nombre!r}. Opciones validas: {opciones}.")
    return constructores[nombre_normalizado](input_shape)


__all__ = [
    "crear_lstm_base",
    "crear_lstm_profunda",
    "crear_lstm_propuesta",
    "crear_modelo",
]
