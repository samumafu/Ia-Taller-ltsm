"""Preparacion cronologica de datos para futuros modelos LSTM.

Este modulo carga el dataset limpio sin modificarlo, codifica la unica variable
categorica, divide cronologicamente, ajusta los escaladores solo con train y
genera ventanas de 12, 24 y 48 horas. No define ni entrena modelos.

Ejecucion desde la raiz del proyecto::

    python -m src.preparacion
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_DATASET = RAIZ_PROYECTO / "data" / "dataset_limpio.csv"
DIRECTORIO_PROCESADOS = RAIZ_PROYECTO / "data" / "processed"
DIRECTORIO_MODELOS = RAIZ_PROYECTO / "models"
RUTA_REPORTE = RAIZ_PROYECTO / "results" / "reporte_preparacion.txt"

COLUMNA_TIEMPO = "timestamp"
TARGET = "demanda_objetivo"
VENTANAS = (12, 24, 48)

FEATURES_ORIGINALES = [
    "demanda_mw",
    "temperatura_c",
    "humedad_pct",
    "viento_kmh",
    "radiacion_wm2",
    "precipitacion_mm",
    "precio_kwh",
    "hora",
    "dia_semana",
    "fin_semana",
    "festivo",
    "mes",
    "periodo_dia",
]

FEATURES_NUMERICAS = [
    "demanda_mw",
    "temperatura_c",
    "humedad_pct",
    "viento_kmh",
    "radiacion_wm2",
    "precipitacion_mm",
    "precio_kwh",
    "hora",
    "dia_semana",
    "fin_semana",
    "festivo",
    "mes",
]

# Orden semantico explicito. Con drop="first", madrugada es la categoria base.
PERIODOS_ORDENADOS = ["madrugada", "mañana", "tarde", "noche"]


@dataclass(frozen=True)
class DivisionTemporal:
    """Indices y timestamps de una division cronologica."""

    nombre: str
    inicio: int
    fin: int
    timestamp_inicial: pd.Timestamp
    timestamp_final: pd.Timestamp

    @property
    def cantidad(self) -> int:
        return self.fin - self.inicio


@dataclass(frozen=True)
class ShapesVentana:
    """Shapes generados para una longitud de ventana."""

    ventana: int
    x_train: tuple[int, ...]
    y_train: tuple[int, ...]
    x_val: tuple[int, ...]
    y_val: tuple[int, ...]
    x_test: tuple[int, ...]
    y_test: tuple[int, ...]


def exigir(condicion: bool, mensaje: str) -> None:
    """Detiene la ejecucion con un mensaje claro si una validacion falla."""

    if not condicion:
        raise ValueError(f"Validacion fallida: {mensaje}")


def cargar_y_validar_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    """Carga el CSV definitivo y valida estructura, tiempo, tipos y target."""

    exigir(RUTA_DATASET.exists(), f"no existe {RUTA_DATASET}")
    datos = pd.read_csv(RUTA_DATASET, encoding="utf-8")

    columnas_requeridas = {COLUMNA_TIEMPO, TARGET, *FEATURES_ORIGINALES}
    faltantes_esquema = sorted(columnas_requeridas - set(datos.columns))
    exigir(not faltantes_esquema, f"faltan columnas: {faltantes_esquema}")

    datos[COLUMNA_TIEMPO] = pd.to_datetime(datos[COLUMNA_TIEMPO], errors="coerce")
    exigir(datos[COLUMNA_TIEMPO].notna().all(), "hay timestamps invalidos")
    exigir(
        datos[COLUMNA_TIEMPO].is_monotonic_increasing,
        "los timestamps no estan ordenados ascendentemente",
    )
    exigir(
        not datos[COLUMNA_TIEMPO].duplicated().any(),
        "hay timestamps duplicados",
    )
    diferencias = datos[COLUMNA_TIEMPO].diff().dropna()
    exigir(
        diferencias.eq(pd.Timedelta(hours=1)).all(),
        "la frecuencia temporal no es horaria y continua",
    )
    exigir(not datos.isna().any().any(), "el dataset contiene NaN")

    for columna in FEATURES_NUMERICAS + [TARGET]:
        exigir(
            pd.api.types.is_numeric_dtype(datos[columna]),
            f"{columna} no tiene tipo numerico",
        )
    exigir(
        pd.api.types.is_string_dtype(datos["periodo_dia"])
        or isinstance(datos["periodo_dia"].dtype, pd.CategoricalDtype),
        "periodo_dia no es categorica o texto",
    )
    categorias = set(datos["periodo_dia"].astype(str).unique())
    exigir(
        categorias.issubset(set(PERIODOS_ORDENADOS)),
        f"periodo_dia contiene categorias inesperadas: {sorted(categorias)}",
    )

    # La ultima etiqueta no puede comprobarse dentro del CSV porque no existe
    # la fila t+1; todas las demas deben coincidir exactamente con shift(-1).
    esperado = datos["demanda_mw"].shift(-1)
    comparables = esperado.notna()
    coincidencias = np.isclose(
        datos.loc[comparables, TARGET].to_numpy(dtype=np.float64),
        esperado.loc[comparables].to_numpy(dtype=np.float64),
        rtol=0,
        atol=1e-9,
    )
    errores_target = int((~coincidencias).sum())
    exigir(errores_target == 0, f"{errores_target} errores en {TARGET}=demanda_mw.shift(-1)")

    validacion_target = {
        "comparables": int(comparables.sum()),
        "correctos": int(coincidencias.sum()),
        "no_verificables_en_csv": int((~comparables).sum()),
        "errores": errores_target,
    }
    return datos, validacion_target


def dividir_cronologicamente(datos: pd.DataFrame) -> dict[str, DivisionTemporal]:
    """Crea cortes contiguos 70/15/15 sin aleatoriedad ni shuffle."""

    cantidad = len(datos)
    cantidad_train = int(cantidad * 0.70)
    cantidad_val = int(cantidad * 0.15)
    corte_val = cantidad_train + cantidad_val

    divisiones = {
        "train": DivisionTemporal(
            "train",
            0,
            cantidad_train,
            datos.at[0, COLUMNA_TIEMPO],
            datos.at[cantidad_train - 1, COLUMNA_TIEMPO],
        ),
        "validation": DivisionTemporal(
            "validation",
            cantidad_train,
            corte_val,
            datos.at[cantidad_train, COLUMNA_TIEMPO],
            datos.at[corte_val - 1, COLUMNA_TIEMPO],
        ),
        "test": DivisionTemporal(
            "test",
            corte_val,
            cantidad,
            datos.at[corte_val, COLUMNA_TIEMPO],
            datos.at[cantidad - 1, COLUMNA_TIEMPO],
        ),
    }

    exigir(
        divisiones["train"].timestamp_final
        < divisiones["validation"].timestamp_inicial,
        "train y validation no estan separados cronologicamente",
    )
    exigir(
        divisiones["validation"].timestamp_final
        < divisiones["test"].timestamp_inicial,
        "validation y test no estan separados cronologicamente",
    )
    exigir(
        sum(division.cantidad for division in divisiones.values()) == cantidad,
        "la division no cubre exactamente todo el dataset",
    )
    return divisiones


def codificar_features(
    datos: pd.DataFrame,
    division_train: DivisionTemporal,
) -> tuple[np.ndarray, list[str], OneHotEncoder]:
    """Ajusta el one-hot solo con train y transforma toda la serie."""

    exigir(TARGET not in FEATURES_ORIGINALES, f"{TARGET} aparece en features")
    exigir(COLUMNA_TIEMPO not in FEATURES_ORIGINALES, "timestamp aparece en features")

    encoder = OneHotEncoder(
        categories=[PERIODOS_ORDENADOS],
        drop="first",
        handle_unknown="error",
        sparse_output=False,
        dtype=np.float64,
    )
    periodo_train = datos.iloc[division_train.inicio : division_train.fin][
        ["periodo_dia"]
    ]
    encoder.fit(periodo_train)
    periodo_codificado = encoder.transform(datos[["periodo_dia"]])
    nombres_one_hot = encoder.get_feature_names_out(["periodo_dia"]).tolist()

    numericas = datos[FEATURES_NUMERICAS].to_numpy(dtype=np.float64)
    matriz = np.concatenate([numericas, periodo_codificado], axis=1)
    features_finales = FEATURES_NUMERICAS + nombres_one_hot

    exigir(matriz.shape[1] == len(features_finales), "numero inconsistente de features")
    exigir(TARGET not in features_finales, f"{TARGET} aparece en X")
    exigir(COLUMNA_TIEMPO not in features_finales, "timestamp aparece en X")
    exigir(np.isfinite(matriz).all(), "X codificado contiene NaN o infinitos")
    exigir(
        len(nombres_one_hot) == len(PERIODOS_ORDENADOS) - 1,
        "one-hot no elimino exactamente una categoria de referencia",
    )
    return matriz, features_finales, encoder


def escalar_sin_fuga(
    x: np.ndarray,
    y: np.ndarray,
    division_train: DivisionTemporal,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """Ajusta ambos scalers exclusivamente con registros de entrenamiento."""

    x_train = x[division_train.inicio : division_train.fin]
    y_train = y[division_train.inicio : division_train.fin]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    scaler_x.fit(x_train)
    scaler_y.fit(y_train)

    exigir(
        int(scaler_x.n_samples_seen_) == division_train.cantidad,
        "scaler_X no fue ajustado exclusivamente con el numero de filas de train",
    )
    exigir(
        int(scaler_y.n_samples_seen_) == division_train.cantidad,
        "scaler_y no fue ajustado exclusivamente con el numero de filas de train",
    )
    exigir(
        np.allclose(scaler_x.mean_, x_train.mean(axis=0), rtol=0, atol=1e-12),
        "la media de scaler_X no coincide con train",
    )
    exigir(
        np.allclose(scaler_y.mean_, y_train.mean(axis=0), rtol=0, atol=1e-12),
        "la media de scaler_y no coincide con train",
    )

    # Validation y test pasan unicamente por transform mediante esta llamada.
    x_escalado = scaler_x.transform(x).astype(np.float32)
    y_escalado = scaler_y.transform(y).astype(np.float32)
    exigir(np.isfinite(x_escalado).all(), "X escalado contiene NaN o infinitos")
    exigir(np.isfinite(y_escalado).all(), "y escalado contiene NaN o infinitos")
    return x_escalado, y_escalado, scaler_x, scaler_y


def crear_secuencias_lstm(
    x_escalado: np.ndarray,
    y_escalado: np.ndarray,
    timestamps: np.ndarray,
    ventana: int,
    inicio_targets: int,
    fin_targets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crea ventanas cuyo ultimo input es t y cuyo target representa t+1.

    ``inicio_targets`` y ``fin_targets`` delimitan a que split pertenece y.
    El contexto X puede comenzar en un split anterior, pero nunca despues del
    timestamp t asociado al target seleccionado.
    """

    exigir(ventana > 0, "la ventana debe ser positiva")
    exigir(inicio_targets >= ventana - 1, "no hay historia suficiente para la ventana")
    exigir(fin_targets <= len(x_escalado), "el rango de targets excede X")
    exigir(fin_targets > inicio_targets, "el rango de targets esta vacio")

    cantidad = fin_targets - inicio_targets
    cantidad_features = x_escalado.shape[1]
    x_secuencias = np.empty(
        (cantidad, ventana, cantidad_features), dtype=np.float32
    )
    y_secuencias = np.empty((cantidad, 1), dtype=np.float32)

    una_hora = np.timedelta64(1, "h")
    for posicion_salida, indice_target in enumerate(range(inicio_targets, fin_targets)):
        inicio_ventana = indice_target - ventana + 1
        fin_ventana = indice_target + 1
        timestamps_entrada = timestamps[inicio_ventana:fin_ventana]
        timestamp_t = timestamps[indice_target]
        timestamp_evento_target = timestamp_t + una_hora

        exigir(
            len(timestamps_entrada) == ventana,
            "una secuencia no tiene la longitud solicitada",
        )
        exigir(
            bool(np.all(timestamps_entrada <= timestamp_t)),
            "una entrada usa una observacion posterior a t",
        )
        exigir(
            bool(np.all(timestamps_entrada < timestamp_evento_target)),
            "una entrada alcanza o supera el timestamp del target t+1",
        )

        x_secuencias[posicion_salida] = x_escalado[inicio_ventana:fin_ventana]
        y_secuencias[posicion_salida] = y_escalado[indice_target]

    exigir(np.isfinite(x_secuencias).all(), "una ventana X contiene NaN o infinitos")
    exigir(np.isfinite(y_secuencias).all(), "una ventana y contiene NaN o infinitos")
    return x_secuencias, y_secuencias


def guardar_artefactos_preprocesamiento(
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    encoder: OneHotEncoder,
    features_finales: list[str],
) -> list[Path]:
    """Persiste transformadores y el orden exacto de las features."""

    DIRECTORIO_MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_scaler_x = DIRECTORIO_MODELOS / "scaler_X.pkl"
    ruta_scaler_y = DIRECTORIO_MODELOS / "scaler_y.pkl"
    ruta_encoder = DIRECTORIO_MODELOS / "encoder_periodo_dia.pkl"
    ruta_features = DIRECTORIO_MODELOS / "features.txt"

    joblib.dump(scaler_x, ruta_scaler_x)
    joblib.dump(scaler_y, ruta_scaler_y)
    joblib.dump(encoder, ruta_encoder)
    ruta_features.write_text("\n".join(features_finales) + "\n", encoding="utf-8")
    return [ruta_scaler_x, ruta_scaler_y, ruta_encoder, ruta_features]


def generar_y_guardar_ventanas(
    x_escalado: np.ndarray,
    y_escalado: np.ndarray,
    timestamps: np.ndarray,
    divisiones: dict[str, DivisionTemporal],
) -> tuple[dict[int, ShapesVentana], list[Path]]:
    """Genera cada ventana, la valida y la guarda antes de liberar memoria."""

    DIRECTORIO_PROCESADOS.mkdir(parents=True, exist_ok=True)
    shapes: dict[int, ShapesVentana] = {}
    rutas: list[Path] = []

    for ventana in VENTANAS:
        x_train, y_train = crear_secuencias_lstm(
            x_escalado,
            y_escalado,
            timestamps,
            ventana,
            ventana - 1,
            divisiones["train"].fin,
        )
        x_val, y_val = crear_secuencias_lstm(
            x_escalado,
            y_escalado,
            timestamps,
            ventana,
            divisiones["validation"].inicio,
            divisiones["validation"].fin,
        )
        x_test, y_test = crear_secuencias_lstm(
            x_escalado,
            y_escalado,
            timestamps,
            ventana,
            divisiones["test"].inicio,
            divisiones["test"].fin,
        )

        for nombre, matriz in {
            "X_train": x_train,
            "X_val": x_val,
            "X_test": x_test,
        }.items():
            exigir(matriz.shape[1] == ventana, f"{nombre}.shape[1] != {ventana}")
            exigir(
                matriz.shape[2] == x_escalado.shape[1],
                f"{nombre} tiene un numero de features diferente",
            )
        exigir(x_train.shape[0] == y_train.shape[0], "X_train/y_train no coinciden")
        exigir(x_val.shape[0] == y_val.shape[0], "X_val/y_val no coinciden")
        exigir(x_test.shape[0] == y_test.shape[0], "X_test/y_test no coinciden")

        ruta = DIRECTORIO_PROCESADOS / f"ventana_{ventana}.npz"
        np.savez_compressed(
            ruta,
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            X_test=x_test,
            y_test=y_test,
        )
        shapes[ventana] = ShapesVentana(
            ventana,
            x_train.shape,
            y_train.shape,
            x_val.shape,
            y_val.shape,
            x_test.shape,
            y_test.shape,
        )
        rutas.append(ruta)
        del x_train, y_train, x_val, y_val, x_test, y_test
        gc.collect()

    return shapes, rutas


def generar_reporte(
    datos: pd.DataFrame,
    validacion_target: dict[str, int],
    features_finales: list[str],
    divisiones: dict[str, DivisionTemporal],
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    shapes: dict[int, ShapesVentana],
) -> None:
    """Genera el reporte de preparacion requerido."""

    lineas = [
        "REPORTE DE PREPARACION DE DATOS PARA LSTM",
        "=" * 80,
        "",
        "1. CARGA Y VALIDACION",
        f"Registros: {len(datos)}",
        f"Rango: {datos[COLUMNA_TIEMPO].min()} a {datos[COLUMNA_TIEMPO].max()}",
        "Timestamp ascendente: True",
        "Timestamps unicos: True",
        "Frecuencia horaria continua: True",
        "NaN en dataset limpio: 0",
        "",
        "2. TARGET",
        f"Target: {TARGET}",
        "Regla: demanda_objetivo[t] = demanda_mw[t+1]",
        f"Relaciones comparables: {validacion_target['comparables']}",
        f"Relaciones correctas: {validacion_target['correctos']}",
        f"Errores: {validacion_target['errores']}",
        (
            "Etiquetas no verificables dentro del CSV por ausencia de t+1: "
            f"{validacion_target['no_verificables_en_csv']}"
        ),
        "demanda_objetivo no aparece en X: True",
        "",
        "3. FEATURES ORIGINALES SELECCIONADAS",
        *[f"  - {feature}" for feature in FEATURES_ORIGINALES],
        "",
        "4. FEATURES FINALES DESPUES DE ONE-HOT",
        "Categoria base eliminada de periodo_dia: madrugada",
        *[f"  {indice + 1:02d}. {feature}" for indice, feature in enumerate(features_finales)],
        f"Cantidad final de features: {len(features_finales)}",
        "",
        "5. DIVISION CRONOLOGICA ANTES DE VENTANAS",
    ]
    for division in divisiones.values():
        lineas.extend(
            [
                (
                    f"{division.nombre}: registros={division.cantidad}, "
                    f"inicio={division.timestamp_inicial}, fin={division.timestamp_final}"
                )
            ]
        )
    lineas.extend(
        [
            (
                "max(timestamp_train) < min(timestamp_validation): "
                f"{divisiones['train'].timestamp_final < divisiones['validation'].timestamp_inicial}"
            ),
            (
                "max(timestamp_validation) < min(timestamp_test): "
                f"{divisiones['validation'].timestamp_final < divisiones['test'].timestamp_inicial}"
            ),
            "",
            "6. NORMALIZACION SIN FUGA",
            (
                "scaler_X.fit y scaler_y.fit ejecutados solo con train: True "
                f"({divisiones['train'].cantidad} registros)"
            ),
            "Validation y test procesados solo con transform: True",
            "Media y desviacion (scale_) de scaler_X calculadas en train:",
        ]
    )
    for feature, media, desviacion in zip(
        features_finales, scaler_x.mean_, scaler_x.scale_, strict=True
    ):
        lineas.append(
            f"  - {feature}: media={media:.12g}, desviacion={desviacion:.12g}"
        )
    lineas.extend(
        [
            (
                f"scaler_y: media={scaler_y.mean_[0]:.12g}, "
                f"desviacion={scaler_y.scale_[0]:.12g}"
            ),
            "",
            "7. SHAPES DE VENTANAS",
        ]
    )
    for ventana, detalle in shapes.items():
        lineas.extend(
            [
                f"Ventana {ventana} horas:",
                f"  X_train={detalle.x_train}; y_train={detalle.y_train}",
                f"  X_val={detalle.x_val}; y_val={detalle.y_val}",
                f"  X_test={detalle.x_test}; y_test={detalle.y_test}",
            ]
        )
    lineas.extend(
        [
            "",
            "8. COMPROBACIONES AUTOMATICAS",
            "Ausencia de NaN en matrices y ventanas: True",
            "Ausencia de infinitos en matrices y ventanas: True",
            "Mismo numero de features en todos los conjuntos: True",
            "Cada X.shape[1] coincide con su ventana: True",
            "Separacion cronologica estricta: True",
            "Ningun input supera el timestamp t de su target t+1: True",
            "Contexto historico previo permitido en validation y test: True",
            f"{TARGET} ausente de X: True",
            "",
            "No se implementaron modelos, entrenamiento, metricas ni graficas.",
        ]
    )
    RUTA_REPORTE.parent.mkdir(parents=True, exist_ok=True)
    RUTA_REPORTE.write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main() -> None:
    """Orquesta la preparacion completa y muestra solo el resumen solicitado."""

    datos, validacion_target = cargar_y_validar_dataset()
    divisiones = dividir_cronologicamente(datos)
    x, features_finales, encoder = codificar_features(datos, divisiones["train"])
    y = datos[[TARGET]].to_numpy(dtype=np.float64)
    x_escalado, y_escalado, scaler_x, scaler_y = escalar_sin_fuga(
        x, y, divisiones["train"]
    )

    artefactos = guardar_artefactos_preprocesamiento(
        scaler_x, scaler_y, encoder, features_finales
    )
    timestamps = datos[COLUMNA_TIEMPO].to_numpy(dtype="datetime64[ns]")
    shapes, archivos_ventanas = generar_y_guardar_ventanas(
        x_escalado, y_escalado, timestamps, divisiones
    )
    generar_reporte(
        datos,
        validacion_target,
        features_finales,
        divisiones,
        scaler_x,
        scaler_y,
        shapes,
    )

    print("RESUMEN DE PREPARACION")
    print(f"1. Features finales: {len(features_finales)}")
    print("2. Tamaños antes de ventanas:")
    for division in divisiones.values():
        print(f"   {division.nombre}: {division.cantidad}")
    print("3. Rangos temporales:")
    for division in divisiones.values():
        print(
            f"   {division.nombre}: {division.timestamp_inicial} a "
            f"{division.timestamp_final}"
        )
    print("4. Shapes:")
    for ventana, detalle in shapes.items():
        print(
            f"   ventana {ventana}: X_train={detalle.x_train}, y_train={detalle.y_train}, "
            f"X_val={detalle.x_val}, y_val={detalle.y_val}, "
            f"X_test={detalle.x_test}, y_test={detalle.y_test}"
        )
    print(
        "5. Scalers: fit SOLO con train "
        f"({divisiones['train'].cantidad} registros); validation/test solo transform."
    )
    print("6. NaN/infinito: 0 en matrices y ventanas.")
    print("7. Archivos generados:")
    for ruta in [*artefactos, *archivos_ventanas, RUTA_REPORTE]:
        print(f"   {ruta.relative_to(RAIZ_PROYECTO)}")


if __name__ == "__main__":
    main()
