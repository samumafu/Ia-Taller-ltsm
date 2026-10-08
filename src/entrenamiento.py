"""Entrenamiento de las nueve combinaciones de modelo y ventana.

Usa exclusivamente los conjuntos train y validation ya preparados. El conjunto
test no se carga ni se evalua en esta etapa.

Ejecucion desde la raiz del proyecto::

    python -m src.entrenamiento
"""

from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tensorflow import keras

from src.modelos import crear_modelo


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIRECTORIO_PROCESADOS = RAIZ_PROYECTO / "data" / "processed"
DIRECTORIO_MODELOS = RAIZ_PROYECTO / "models"
DIRECTORIO_RESULTADOS = RAIZ_PROYECTO / "results"
RUTA_HISTORIALES = DIRECTORIO_RESULTADOS / "historiales.csv"
RUTA_RESUMEN = DIRECTORIO_RESULTADOS / "resumen_entrenamiento.csv"

NOMBRES_MODELOS = ("base", "profunda", "propuesta")
VENTANAS = (12, 24, 48)
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10
SEMILLA = 2026


def exigir(condicion: bool, mensaje: str) -> None:
    """Detiene el entrenamiento si una precondicion no se cumple."""

    if not condicion:
        raise ValueError(f"Validacion fallida: {mensaje}")


class ProgresoExperimento(keras.callbacks.Callback):
    """Muestra una linea compacta por epoca para seguir ejecuciones largas."""

    def __init__(self, nombre: str, ventana: int) -> None:
        super().__init__()
        self.nombre = nombre
        self.ventana = ventana

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        valores = logs or {}
        loss = float(valores.get("loss", np.nan))
        val_loss = float(valores.get("val_loss", np.nan))
        print(
            f"[{self.nombre} v{self.ventana}] epoca {epoch + 1:03d}/{EPOCHS} "
            f"loss={loss:.6f} val_loss={val_loss:.6f}",
            flush=True,
        )


def cargar_train_validation(
    ventana: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga solo train y validation del NPZ de una ventana."""

    ruta = DIRECTORIO_PROCESADOS / f"ventana_{ventana}.npz"
    exigir(ruta.exists(), f"no existe {ruta}")

    with np.load(ruta) as datos:
        claves_necesarias = {"X_train", "y_train", "X_val", "y_val"}
        exigir(
            claves_necesarias.issubset(datos.files),
            f"{ruta.name} no contiene {sorted(claves_necesarias)}",
        )
        # No se accede a X_test ni y_test durante el entrenamiento.
        x_train = datos["X_train"]
        y_train = datos["y_train"]
        x_val = datos["X_val"]
        y_val = datos["y_val"]

    for nombre, arreglo in {
        "X_train": x_train,
        "y_train": y_train,
        "X_val": x_val,
        "y_val": y_val,
    }.items():
        exigir(np.isfinite(arreglo).all(), f"{nombre} contiene NaN o infinitos")

    exigir(x_train.ndim == 3 and x_val.ndim == 3, "X debe tener tres dimensiones")
    exigir(y_train.ndim == 2 and y_val.ndim == 2, "y debe tener dos dimensiones")
    exigir(x_train.shape[1] == ventana, "X_train no coincide con la ventana")
    exigir(x_val.shape[1] == ventana, "X_val no coincide con la ventana")
    exigir(x_train.shape[2] == x_val.shape[2], "train/validation difieren en features")
    exigir(x_train.shape[0] == y_train.shape[0], "X_train/y_train no coinciden")
    exigir(x_val.shape[0] == y_val.shape[0], "X_val/y_val no coinciden")
    exigir(y_train.shape[1] == 1 and y_val.shape[1] == 1, "y debe tener shape (*, 1)")
    return x_train, y_train, x_val, y_val


def guardar_resultados_parciales(
    historiales: list[dict[str, str | int | float]],
    resumen: list[dict[str, str | int | float]],
) -> None:
    """Persiste resultados despues de cada experimento completado."""

    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        historiales,
        columns=["modelo", "ventana", "epoca", "loss", "val_loss"],
    ).to_csv(RUTA_HISTORIALES, index=False, encoding="utf-8")
    pd.DataFrame(
        resumen,
        columns=[
            "modelo",
            "ventana",
            "epocas",
            "tiempo_segundos",
            "mejor_val_loss",
        ],
    ).to_csv(RUTA_RESUMEN, index=False, encoding="utf-8")


def entrenar_experimento(
    nombre_modelo: str,
    ventana: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    indice_experimento: int,
) -> tuple[list[dict[str, str | int | float]], dict[str, str | int | float]]:
    """Entrena, restaura los mejores pesos y guarda una combinacion."""

    keras.backend.clear_session()
    keras.utils.set_random_seed(SEMILLA + indice_experimento)

    input_shape = (ventana, x_train.shape[2])
    modelo = crear_modelo(nombre_modelo, input_shape)
    exigir(modelo.input_shape[1:] == input_shape, "input_shape incorrecto en el modelo")
    exigir(modelo.output_shape[-1] == 1, "la salida del modelo no es escalar")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )
    progreso = ProgresoExperimento(nombre_modelo, ventana)

    print(
        f"\nIniciando experimento {indice_experimento}/9: "
        f"modelo={nombre_modelo}, ventana={ventana}, input_shape={input_shape}",
        flush=True,
    )
    inicio = perf_counter()
    historia = modelo.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, progreso],
        shuffle=False,
        verbose=0,
    )
    tiempo_segundos = perf_counter() - inicio

    losses = [float(valor) for valor in historia.history["loss"]]
    val_losses = [float(valor) for valor in historia.history["val_loss"]]
    exigir(len(losses) == len(val_losses), "loss y val_loss difieren en longitud")
    exigir(len(losses) > 0, "el entrenamiento no ejecuto ninguna epoca")
    exigir(
        np.isfinite(losses).all() and np.isfinite(val_losses).all(),
        "el historial contiene NaN o infinitos",
    )

    epocas_ejecutadas = len(losses)
    mejor_val_loss = min(val_losses)
    DIRECTORIO_MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_modelo = DIRECTORIO_MODELOS / f"{nombre_modelo}_v{ventana}.keras"
    modelo.save(ruta_modelo)

    filas_historial = [
        {
            "modelo": nombre_modelo,
            "ventana": ventana,
            "epoca": epoca,
            "loss": loss,
            "val_loss": val_loss,
        }
        for epoca, (loss, val_loss) in enumerate(zip(losses, val_losses, strict=True), start=1)
    ]
    fila_resumen: dict[str, str | int | float] = {
        "modelo": nombre_modelo,
        "ventana": ventana,
        "epocas": epocas_ejecutadas,
        "tiempo_segundos": tiempo_segundos,
        "mejor_val_loss": mejor_val_loss,
    }
    print(
        f"Finalizado: {nombre_modelo} v{ventana} | epocas={epocas_ejecutadas} | "
        f"tiempo={tiempo_segundos:.2f}s | mejor_val_loss={mejor_val_loss:.6f} | "
        f"guardado={ruta_modelo.name}",
        flush=True,
    )
    return filas_historial, fila_resumen


def main() -> None:
    """Ejecuta los nueve experimentos y genera los dos CSV solicitados."""

    historiales: list[dict[str, str | int | float]] = []
    resumen: list[dict[str, str | int | float]] = []
    indice_experimento = 0

    for ventana in VENTANAS:
        x_train, y_train, x_val, y_val = cargar_train_validation(ventana)
        for nombre_modelo in NOMBRES_MODELOS:
            indice_experimento += 1
            filas_historial, fila_resumen = entrenar_experimento(
                nombre_modelo,
                ventana,
                x_train,
                y_train,
                x_val,
                y_val,
                indice_experimento,
            )
            historiales.extend(filas_historial)
            resumen.append(fila_resumen)
            guardar_resultados_parciales(historiales, resumen)
            keras.backend.clear_session()
            gc.collect()

        del x_train, y_train, x_val, y_val
        gc.collect()

    exigir(len(resumen) == 9, "no se completaron los nueve experimentos")
    print("\nRESUMEN FINAL DE ENTRENAMIENTO", flush=True)
    print(pd.DataFrame(resumen).to_string(index=False), flush=True)
    print(f"\nHistoriales: {RUTA_HISTORIALES}", flush=True)
    print(f"Resumen: {RUTA_RESUMEN}", flush=True)
    print("No se calcularon metricas de test ni se generaron graficas.", flush=True)


if __name__ == "__main__":
    main()
