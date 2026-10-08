"""Generacion de graficas a partir de resultados ya calculados.

Este modulo no carga modelos, no predice, no evalua y no entrena. Solo lee los
CSV existentes y guarda figuras PNG a 300 dpi.

Ejecucion futura desde la raiz del proyecto::

    python -m src.graficas
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Permite guardar figuras sin requerir una ventana o sesion grafica.
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
DIRECTORIO_RESULTADOS = RAIZ_PROYECTO / "results"
DIRECTORIO_PREDICCIONES = DIRECTORIO_RESULTADOS / "predicciones"
DIRECTORIO_GRAFICAS = DIRECTORIO_RESULTADOS / "graficas"
RUTA_HISTORIALES = DIRECTORIO_RESULTADOS / "historiales.csv"
RUTA_RESULTADOS_MODELOS = DIRECTORIO_RESULTADOS / "resultados_modelos.csv"

MODELOS = ("base", "profunda", "propuesta")
VENTANAS = (12, 24, 48)
DPI = 300

NOMBRES_MODELOS = {
    "base": "LSTM base",
    "profunda": "LSTM profunda",
    "propuesta": "LSTM propuesta",
}

COLORES_MODELOS = {
    "base": "#2878B5",
    "profunda": "#F28E2B",
    "propuesta": "#59A14F",
}


def exigir(condicion: bool, mensaje: str) -> None:
    """Detiene el script con un mensaje claro ante datos incompletos."""

    if not condicion:
        raise ValueError(f"Validacion fallida: {mensaje}")


def combinaciones_esperadas() -> set[tuple[str, int]]:
    """Retorna las nueve combinaciones modelo/ventana requeridas."""

    return {(modelo, ventana) for modelo in MODELOS for ventana in VENTANAS}


def validar_combinaciones(datos: pd.DataFrame, origen: str) -> None:
    """Comprueba que un DataFrame represente exactamente los 9 experimentos."""

    encontradas = set(
        datos[["modelo", "ventana"]].itertuples(index=False, name=None)
    )
    exigir(
        encontradas == combinaciones_esperadas(),
        f"{origen} no contiene exactamente las nueve combinaciones esperadas",
    )


def guardar_figura(figura: plt.Figure, ruta: Path) -> None:
    """Guarda una figura con configuración homogénea y libera memoria."""

    figura.tight_layout()
    figura.savefig(
        ruta,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figura)


def configurar_eje_temporal(eje: plt.Axes) -> None:
    """Mantiene legibles los timestamps en series extensas."""

    localizador = mdates.AutoDateLocator(minticks=4, maxticks=9)
    eje.xaxis.set_major_locator(localizador)
    eje.xaxis.set_major_formatter(mdates.ConciseDateFormatter(localizador))


def cargar_historiales() -> pd.DataFrame:
    """Carga y valida los historiales de entrenamiento/validación."""

    exigir(RUTA_HISTORIALES.exists(), f"no existe {RUTA_HISTORIALES}")
    historiales = pd.read_csv(RUTA_HISTORIALES, encoding="utf-8")
    columnas = {"modelo", "ventana", "epoca", "loss", "val_loss"}
    exigir(columnas.issubset(historiales.columns), "historiales.csv tiene columnas incompletas")
    exigir(not historiales.empty, "historiales.csv está vacío")
    exigir(
        np.isfinite(historiales[["epoca", "loss", "val_loss"]]).all().all(),
        "historiales.csv contiene NaN o infinitos",
    )
    validar_combinaciones(historiales, "historiales.csv")
    return historiales


def generar_graficas_loss(historiales: pd.DataFrame) -> list[Path]:
    """Genera loss de entrenamiento frente a validación para cada experimento."""

    rutas: list[Path] = []
    for ventana in VENTANAS:
        for modelo in MODELOS:
            datos = historiales.loc[
                (historiales["modelo"] == modelo)
                & (historiales["ventana"] == ventana)
            ].sort_values("epoca")
            exigir(not datos.empty, f"falta historial para {modelo} v{ventana}")
            exigir(
                datos["epoca"].tolist() == list(range(1, len(datos) + 1)),
                f"las épocas de {modelo} v{ventana} no son consecutivas",
            )

            figura, eje = plt.subplots(figsize=(10, 6))
            eje.plot(
                datos["epoca"],
                datos["loss"],
                color="#2878B5",
                linewidth=2,
                marker="o",
                markersize=3,
                label="Entrenamiento",
            )
            eje.plot(
                datos["epoca"],
                datos["val_loss"],
                color="#D95319",
                linewidth=2,
                marker="o",
                markersize=3,
                label="Validación",
            )
            eje.set_title(
                f"Loss de entrenamiento y validación — {NOMBRES_MODELOS[modelo]}, "
                f"ventana {ventana} h"
            )
            eje.set_xlabel("Época")
            eje.set_ylabel("MSE normalizado")
            eje.grid(True, alpha=0.3)
            eje.legend(title="Serie")

            ruta = DIRECTORIO_GRAFICAS / f"loss_{modelo}_v{ventana}.png"
            guardar_figura(figura, ruta)
            rutas.append(ruta)
    return rutas


def cargar_predicciones(modelo: str, ventana: int) -> pd.DataFrame:
    """Carga un CSV de predicciones ya calculadas y valida su esquema."""

    ruta = DIRECTORIO_PREDICCIONES / f"{modelo}_v{ventana}.csv"
    exigir(ruta.exists(), f"no existe {ruta}")
    datos = pd.read_csv(ruta, encoding="utf-8")
    columnas = {"timestamp_objetivo", "y_real_mw", "y_pred_mw"}
    exigir(columnas.issubset(datos.columns), f"{ruta.name} tiene columnas incompletas")
    datos["timestamp_objetivo"] = pd.to_datetime(
        datos["timestamp_objetivo"], errors="coerce"
    )
    exigir(datos["timestamp_objetivo"].notna().all(), f"{ruta.name} tiene timestamps inválidos")
    exigir(
        datos["timestamp_objetivo"].is_monotonic_increasing,
        f"{ruta.name} no está ordenado temporalmente",
    )
    exigir(
        np.isfinite(datos[["y_real_mw", "y_pred_mw"]]).all().all(),
        f"{ruta.name} contiene NaN o infinitos",
    )
    exigir(not datos.empty, f"{ruta.name} está vacío")
    return datos


def generar_graficas_prediccion_y_error() -> list[Path]:
    """Genera demanda real/predicha y error real-predicha para cada modelo."""

    rutas: list[Path] = []
    for ventana in VENTANAS:
        for modelo in MODELOS:
            datos = cargar_predicciones(modelo, ventana)
            tiempos = datos["timestamp_objetivo"]
            y_real = datos["y_real_mw"]
            y_pred = datos["y_pred_mw"]

            figura_pred, eje_pred = plt.subplots(figsize=(14, 6))
            eje_pred.plot(
                tiempos,
                y_real,
                color="#202020",
                linewidth=1.2,
                label="Demanda real",
            )
            eje_pred.plot(
                tiempos,
                y_pred,
                color=COLORES_MODELOS[modelo],
                linewidth=1.1,
                alpha=0.9,
                label="Demanda predicha",
            )
            eje_pred.set_title(
                f"Demanda real vs. predicha — {NOMBRES_MODELOS[modelo]}, "
                f"ventana {ventana} h"
            )
            eje_pred.set_xlabel("Timestamp objetivo")
            eje_pred.set_ylabel("Demanda (MW)")
            eje_pred.grid(True, alpha=0.25)
            eje_pred.legend(title="Serie")
            configurar_eje_temporal(eje_pred)
            ruta_pred = DIRECTORIO_GRAFICAS / f"prediccion_{modelo}_v{ventana}.png"
            guardar_figura(figura_pred, ruta_pred)
            rutas.append(ruta_pred)

            # La definición solicitada es real - predicha. No se reutiliza la
            # columna error_mw porque evaluación la guarda con el signo opuesto.
            error = y_real - y_pred
            figura_error, eje_error = plt.subplots(figsize=(14, 5.5))
            eje_error.plot(
                tiempos,
                error,
                color="#C44E52",
                linewidth=1,
                label="Error (real − predicha)",
            )
            eje_error.axhline(
                0,
                color="#202020",
                linestyle="--",
                linewidth=1,
                label="Error cero",
            )
            eje_error.set_title(
                f"Error de predicción — {NOMBRES_MODELOS[modelo]}, "
                f"ventana {ventana} h"
            )
            eje_error.set_xlabel("Timestamp objetivo")
            eje_error.set_ylabel("Error (MW)")
            eje_error.grid(True, alpha=0.25)
            eje_error.legend(title="Referencia")
            configurar_eje_temporal(eje_error)
            ruta_error = DIRECTORIO_GRAFICAS / f"error_{modelo}_v{ventana}.png"
            guardar_figura(figura_error, ruta_error)
            rutas.append(ruta_error)
    return rutas


def cargar_resultados_modelos() -> pd.DataFrame:
    """Carga métricas y normaliza nombres para su validación interna."""

    exigir(RUTA_RESULTADOS_MODELOS.exists(), f"no existe {RUTA_RESULTADOS_MODELOS}")
    resultados = pd.read_csv(RUTA_RESULTADOS_MODELOS, encoding="utf-8")
    columnas = {"Modelo", "Ventana", "MAE", "RMSE", "MAPE", "R²"}
    exigir(columnas.issubset(resultados.columns), "resultados_modelos.csv tiene columnas incompletas")
    exigir(
        np.isfinite(resultados[["Ventana", "MAE", "RMSE", "MAPE", "R²"]]).all().all(),
        "resultados_modelos.csv contiene NaN o infinitos",
    )
    exigir(
        not resultados.duplicated(subset=["Modelo", "Ventana"]).any(),
        "resultados_modelos.csv tiene combinaciones duplicadas",
    )
    resultados = resultados.rename(columns={"Modelo": "modelo", "Ventana": "ventana"})
    validar_combinaciones(resultados, "resultados_modelos.csv")
    return resultados


def generar_comparacion_metricas(resultados: pd.DataFrame) -> Path:
    """Compara los nueve experimentos en cuatro subgráficas de barras."""

    orden_modelos = {modelo: posicion for posicion, modelo in enumerate(MODELOS)}
    resultados = resultados.assign(
        orden_modelo=resultados["modelo"].map(orden_modelos)
    ).sort_values(["ventana", "orden_modelo"])
    etiquetas = [
        f"{NOMBRES_MODELOS[modelo]}\nv{ventana}"
        for modelo, ventana in resultados[["modelo", "ventana"]].itertuples(
            index=False, name=None
        )
    ]
    colores = [COLORES_MODELOS[modelo] for modelo in resultados["modelo"]]
    configuracion = [
        ("MAE", "MAE (MW)", "MAE — menor es mejor"),
        ("RMSE", "RMSE (MW)", "RMSE — menor es mejor"),
        ("MAPE", "MAPE (%)", "MAPE — menor es mejor"),
        ("R²", "R²", "R² — mayor es mejor"),
    ]

    figura, ejes = plt.subplots(2, 2, figsize=(18, 12))
    posiciones = np.arange(len(resultados))
    for eje, (metrica, etiqueta_y, titulo) in zip(
        ejes.flat, configuracion, strict=True
    ):
        barras = eje.bar(
            posiciones,
            resultados[metrica],
            color=colores,
            edgecolor="white",
            linewidth=0.8,
        )
        eje.set_title(titulo)
        eje.set_ylabel(etiqueta_y)
        eje.set_xticks(posiciones, etiquetas, rotation=35, ha="right")
        eje.grid(axis="y", alpha=0.25)
        formato = "%.3f" if metrica == "R²" else "%.2f"
        eje.bar_label(barras, fmt=formato, padding=3, fontsize=8)

    leyenda = [
        Patch(color=COLORES_MODELOS[modelo], label=NOMBRES_MODELOS[modelo])
        for modelo in MODELOS
    ]
    figura.legend(
        handles=leyenda,
        loc="upper center",
        ncol=3,
        title="Arquitectura",
        bbox_to_anchor=(0.5, 1.01),
    )
    figura.suptitle(
        "Comparación de métricas de los 9 modelos sobre el conjunto de prueba",
        fontsize=16,
        y=1.055,
    )
    ruta = DIRECTORIO_GRAFICAS / "comparacion_modelos_metricas.png"
    guardar_figura(figura, ruta)
    return ruta


def main() -> None:
    """Genera las 28 imágenes solicitadas exclusivamente desde CSV existentes."""

    DIRECTORIO_GRAFICAS.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    historiales = cargar_historiales()
    rutas_loss = generar_graficas_loss(historiales)
    rutas_series = generar_graficas_prediccion_y_error()
    resultados = cargar_resultados_modelos()
    ruta_comparacion = generar_comparacion_metricas(resultados)
    rutas = [*rutas_loss, *rutas_series, ruta_comparacion]

    exigir(len(rutas_loss) == 9, "no se generaron las nueve gráficas de loss")
    exigir(len(rutas_series) == 18, "no se generaron las 18 gráficas de series/error")
    exigir(len(rutas) == 28, "el total de gráficas generadas no es 28")
    exigir(all(ruta.exists() for ruta in rutas), "algún PNG no fue guardado")

    print(f"Gráficas generadas: {len(rutas)}")
    print(f"Directorio: {DIRECTORIO_GRAFICAS}")
    print("No se entrenaron ni evaluaron modelos.")


if __name__ == "__main__":
    main()
