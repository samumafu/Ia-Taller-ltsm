"""Evaluacion de los nueve modelos LSTM sobre sus conjuntos de prueba.

El modulo carga modelos ya entrenados; no compila arquitecturas, no reentrena y
no genera graficas. Las metricas se calculan en MW despues de invertir el
escalado del target.

Ejecucion futura desde la raiz del proyecto::

    python -m src.evaluacion
"""

from __future__ import annotations

import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from tensorflow import keras


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIRECTORIO_PROCESADOS = RAIZ_PROYECTO / "data" / "processed"
DIRECTORIO_MODELOS = RAIZ_PROYECTO / "models"
DIRECTORIO_RESULTADOS = RAIZ_PROYECTO / "results"
DIRECTORIO_PREDICCIONES = DIRECTORIO_RESULTADOS / "predicciones"

RUTA_DATASET_LIMPIO = RAIZ_PROYECTO / "data" / "dataset_limpio.csv"
RUTA_SCALER_Y = DIRECTORIO_MODELOS / "scaler_y.pkl"
RUTA_RESUMEN_ENTRENAMIENTO = DIRECTORIO_RESULTADOS / "resumen_entrenamiento.csv"
RUTA_RESULTADOS = DIRECTORIO_RESULTADOS / "resultados_modelos.csv"

MODELOS_VALIDOS = ("base", "profunda", "propuesta")
VENTANAS_VALIDAS = (12, 24, 48)
COLUMNAS_RESULTADOS = [
    "Modelo",
    "Ventana",
    "MAE",
    "MSE",
    "RMSE",
    "MAPE",
    "R²",
    "Épocas",
]


def exigir(condicion: bool, mensaje: str) -> None:
    """Detiene la evaluacion con un mensaje explicito si algo no es valido."""

    if not condicion:
        raise ValueError(f"Validacion fallida: {mensaje}")


def cargar_resumen_entrenamiento() -> pd.DataFrame:
    """Carga las epocas realmente ejecutadas y valida los nueve experimentos."""

    exigir(
        RUTA_RESUMEN_ENTRENAMIENTO.exists(),
        f"no existe {RUTA_RESUMEN_ENTRENAMIENTO}",
    )
    resumen = pd.read_csv(RUTA_RESUMEN_ENTRENAMIENTO, encoding="utf-8")
    columnas_requeridas = {"modelo", "ventana", "epocas"}
    exigir(
        columnas_requeridas.issubset(resumen.columns),
        f"el resumen no contiene {sorted(columnas_requeridas)}",
    )
    exigir(
        not resumen.duplicated(subset=["modelo", "ventana"]).any(),
        "hay combinaciones modelo/ventana duplicadas en el resumen",
    )

    combinaciones_esperadas = {
        (modelo, ventana)
        for modelo in MODELOS_VALIDOS
        for ventana in VENTANAS_VALIDAS
    }
    combinaciones_encontradas = set(
        resumen[["modelo", "ventana"]].itertuples(index=False, name=None)
    )
    exigir(
        combinaciones_encontradas == combinaciones_esperadas,
        "el resumen no contiene exactamente los nueve experimentos esperados",
    )
    exigir((resumen["epocas"] > 0).all(), "hay cantidades de epocas no positivas")
    return resumen


def cargar_timestamps_objetivo_test(cantidad_test: int) -> pd.Series:
    """Reconstruye el timestamp t+1 asociado a cada fila del conjunto test.

    La preparacion usa cortes cronologicos floor(70 %) y floor(15 %); test recibe
    el remanente. Los timestamps se guardan solo como metadatos para graficas
    futuras y no intervienen en las metricas.
    """

    exigir(RUTA_DATASET_LIMPIO.exists(), f"no existe {RUTA_DATASET_LIMPIO}")
    tiempos = pd.read_csv(
        RUTA_DATASET_LIMPIO,
        usecols=["timestamp"],
        encoding="utf-8",
    )
    tiempos["timestamp"] = pd.to_datetime(tiempos["timestamp"], errors="coerce")
    exigir(tiempos["timestamp"].notna().all(), "hay timestamps invalidos")
    exigir(tiempos["timestamp"].is_monotonic_increasing, "timestamps no ordenados")

    cantidad_total = len(tiempos)
    cantidad_train = int(cantidad_total * 0.70)
    cantidad_val = int(cantidad_total * 0.15)
    inicio_test = cantidad_train + cantidad_val
    timestamps_contexto = tiempos.loc[inicio_test:, "timestamp"].reset_index(drop=True)
    exigir(
        len(timestamps_contexto) == cantidad_test,
        "el numero de timestamps de test no coincide con X_test/y_test",
    )
    return timestamps_contexto + pd.Timedelta(hours=1)


def cargar_test(ventana: int) -> tuple[np.ndarray, np.ndarray]:
    """Carga y valida X_test/y_test de una ventana preparada."""

    ruta_npz = DIRECTORIO_PROCESADOS / f"ventana_{ventana}.npz"
    exigir(ruta_npz.exists(), f"no existe {ruta_npz}")
    with np.load(ruta_npz) as datos:
        exigir(
            {"X_test", "y_test"}.issubset(datos.files),
            f"{ruta_npz.name} no contiene X_test/y_test",
        )
        x_test = datos["X_test"]
        y_test = datos["y_test"]

    exigir(x_test.ndim == 3, "X_test debe tener tres dimensiones")
    exigir(y_test.ndim == 2 and y_test.shape[1] == 1, "y_test debe tener shape (*, 1)")
    exigir(x_test.shape[0] == y_test.shape[0], "X_test/y_test no coinciden")
    exigir(x_test.shape[1] == ventana, "X_test no coincide con su ventana")
    exigir(np.isfinite(x_test).all(), "X_test contiene NaN o infinitos")
    exigir(np.isfinite(y_test).all(), "y_test contiene NaN o infinitos")
    return x_test, y_test


def calcular_metricas(
    y_real_mw: np.ndarray,
    y_pred_mw: np.ndarray,
) -> dict[str, float]:
    """Calcula metricas de regresion en la escala original de MW.

    MAPE se expresa como porcentaje, no como proporcion.
    """

    y_real = np.asarray(y_real_mw, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred_mw, dtype=np.float64).reshape(-1)
    exigir(y_real.shape == y_pred.shape, "predicciones y valores reales no coinciden")
    exigir(np.isfinite(y_real).all(), "y real en MW contiene NaN o infinitos")
    exigir(np.isfinite(y_pred).all(), "predicciones en MW contienen NaN o infinitos")
    exigir(not np.isclose(y_real, 0).any(), "MAPE no esta definida porque y real contiene cero")

    mse = float(mean_squared_error(y_real, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(mean_absolute_percentage_error(y_real, y_pred) * 100.0),
        "R²": float(r2_score(y_real, y_pred)),
    }


def evaluar_experimento(
    modelo_nombre: str,
    ventana: int,
    epocas: int,
    scaler_y: object,
    timestamps_objetivo: pd.Series,
) -> dict[str, str | int | float]:
    """Carga un modelo, predice test, invierte escala y guarda predicciones."""

    x_test, y_test_escalado = cargar_test(ventana)
    exigir(
        len(timestamps_objetivo) == len(y_test_escalado),
        "timestamps y targets de test no coinciden",
    )

    ruta_modelo = DIRECTORIO_MODELOS / f"{modelo_nombre}_v{ventana}.keras"
    exigir(ruta_modelo.exists(), f"no existe {ruta_modelo}")
    modelo = keras.models.load_model(ruta_modelo, compile=False)
    exigir(
        tuple(modelo.input_shape[1:]) == tuple(x_test.shape[1:]),
        f"el input del modelo {ruta_modelo.name} no coincide con X_test",
    )
    exigir(modelo.output_shape[-1] == 1, "la salida del modelo no es escalar")

    y_pred_escalado = modelo.predict(x_test, verbose=0)
    exigir(
        y_pred_escalado.shape == y_test_escalado.shape,
        "la forma de las predicciones no coincide con y_test",
    )

    y_real_mw = scaler_y.inverse_transform(y_test_escalado).reshape(-1)
    y_pred_mw = scaler_y.inverse_transform(y_pred_escalado).reshape(-1)
    metricas = calcular_metricas(y_real_mw, y_pred_mw)

    predicciones = pd.DataFrame(
        {
            "timestamp_objetivo": timestamps_objetivo,
            "y_real_mw": y_real_mw,
            "y_pred_mw": y_pred_mw,
            "error_mw": y_pred_mw - y_real_mw,
            "error_absoluto_mw": np.abs(y_pred_mw - y_real_mw),
        }
    )
    DIRECTORIO_PREDICCIONES.mkdir(parents=True, exist_ok=True)
    ruta_predicciones = (
        DIRECTORIO_PREDICCIONES / f"{modelo_nombre}_v{ventana}.csv"
    )
    predicciones.to_csv(
        ruta_predicciones,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    return {
        "Modelo": modelo_nombre,
        "Ventana": ventana,
        **metricas,
        "Épocas": epocas,
    }


def main() -> None:
    """Evalua las nueve combinaciones y genera resultados reproducibles."""

    exigir(RUTA_SCALER_Y.exists(), f"no existe {RUTA_SCALER_Y}")
    scaler_y = joblib.load(RUTA_SCALER_Y)
    exigir(hasattr(scaler_y, "inverse_transform"), "scaler_y no es valido")
    resumen_entrenamiento = cargar_resumen_entrenamiento()

    # Las tres ventanas comparten los mismos targets de test; se toma su tamaño
    # de un artefacto preparado y luego se valida de nuevo en cada experimento.
    _, y_test_referencia = cargar_test(VENTANAS_VALIDAS[0])
    timestamps_objetivo = cargar_timestamps_objetivo_test(len(y_test_referencia))

    resultados: list[dict[str, str | int | float]] = []
    for fila in resumen_entrenamiento.itertuples(index=False):
        resultado = evaluar_experimento(
            modelo_nombre=str(fila.modelo),
            ventana=int(fila.ventana),
            epocas=int(fila.epocas),
            scaler_y=scaler_y,
            timestamps_objetivo=timestamps_objetivo,
        )
        resultados.append(resultado)
        print(
            f"Evaluado {fila.modelo} v{fila.ventana}: "
            f"MAE={resultado['MAE']:.4f} MW, RMSE={resultado['RMSE']:.4f} MW",
            flush=True,
        )
        keras.backend.clear_session()
        gc.collect()

    exigir(len(resultados) == 9, "no se evaluaron los nueve modelos")
    tabla_resultados = pd.DataFrame(resultados, columns=COLUMNAS_RESULTADOS)
    exigir(not tabla_resultados.isna().any().any(), "las metricas contienen NaN")
    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabla_resultados.to_csv(RUTA_RESULTADOS, index=False, encoding="utf-8")

    print("\nRESULTADOS DE TEST EN MW")
    print(tabla_resultados.to_string(index=False))
    print(f"\nResultados: {RUTA_RESULTADOS}")
    print(f"Predicciones: {DIRECTORIO_PREDICCIONES}")
    print("No se reentrenaron modelos ni se generaron graficas.")


if __name__ == "__main__":
    main()
