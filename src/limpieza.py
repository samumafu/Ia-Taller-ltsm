"""Limpieza definitiva y auditable del dataset horario de demanda energetica.

No normaliza, divide datos, crea ventanas ni entrena modelos. Se ejecuta desde
la raiz del proyecto con: ``python -m src.limpieza``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_ENTRADA = RAIZ_PROYECTO / "data" / "dataset_demanda_energia_LSTM_2025_2026.csv"
RUTA_SALIDA = RAIZ_PROYECTO / "data" / "dataset_limpio.csv"
RUTA_REPORTE = RAIZ_PROYECTO / "results" / "reporte_limpieza.txt"
COLUMNA_TIEMPO = "timestamp"
MAX_HORAS_INTERPOLABLES = 2

COLUMNAS_INTERPOLABLES = [
    "demanda_mw",
    "temperatura_c",
    "humedad_pct",
    "viento_kmh",
    "radiacion_wm2",
    "precipitacion_mm",
    "precio_kwh",
]

COLUMNAS_REQUERIDAS_MODELO = [
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
    "demanda_objetivo",
]


@dataclass(frozen=True)
class ResultadoInterpolacion:
    """Resumen de la interpolacion aplicada a una columna."""

    faltantes_antes: int
    huecos_detectados: int
    longitud_maxima: int
    valores_interpolados: int
    faltantes_despues: int
    timestamps_interpolados: tuple[pd.Timestamp, ...]


def titulo(texto: str) -> None:
    print(f"\n{'=' * 80}\n{texto}\n{'=' * 80}")


def conteos_faltantes(datos: pd.DataFrame) -> dict[str, int]:
    """Retorna los conteos de NaN de todas las columnas."""

    return {columna: int(cantidad) for columna, cantidad in datos.isna().sum().items()}


def corridas_faltantes(datos: pd.DataFrame, columna: str) -> list[tuple[int, int]]:
    """Localiza posiciones inicial/final de cada corrida consecutiva de NaN."""

    mascara = datos[columna].isna().to_numpy()
    corridas: list[tuple[int, int]] = []
    posicion = 0
    while posicion < len(datos):
        if not mascara[posicion]:
            posicion += 1
            continue
        inicio = posicion
        while posicion + 1 < len(datos) and mascara[posicion + 1]:
            posicion += 1
        corridas.append((inicio, posicion))
        posicion += 1
    return corridas


def interpolar_huecos_locales(
    datos: pd.DataFrame,
    columna: str,
    max_horas: int = MAX_HORAS_INTERPOLABLES,
) -> ResultadoInterpolacion:
    """Interpola solo huecos cortos, interiores y temporalmente continuos.

    Cada valor se obtiene de los extremos inmediatos mediante interpolacion
    lineal respecto del timestamp. No se usan estadisticas globales y nunca se
    extrapolan el inicio ni el final de la serie.
    """

    corridas = corridas_faltantes(datos, columna)
    faltantes_antes = int(datos[columna].isna().sum())
    longitudes = [fin - inicio + 1 for inicio, fin in corridas]
    interpolados: list[pd.Timestamp] = []

    for inicio, fin in corridas:
        longitud = fin - inicio + 1
        anterior = inicio - 1
        siguiente = fin + 1
        if longitud > max_horas or anterior < 0 or siguiente >= len(datos):
            continue

        valor_anterior = datos.at[anterior, columna]
        valor_siguiente = datos.at[siguiente, columna]
        if pd.isna(valor_anterior) or pd.isna(valor_siguiente):
            continue

        tiempo_anterior = datos.at[anterior, COLUMNA_TIEMPO]
        tiempo_siguiente = datos.at[siguiente, COLUMNA_TIEMPO]
        if tiempo_siguiente - tiempo_anterior != pd.Timedelta(hours=longitud + 1):
            continue

        duracion = (tiempo_siguiente - tiempo_anterior).total_seconds()
        for posicion in range(inicio, fin + 1):
            tiempo_actual = datos.at[posicion, COLUMNA_TIEMPO]
            proporcion = (tiempo_actual - tiempo_anterior).total_seconds() / duracion
            datos.at[posicion, columna] = valor_anterior + (
                valor_siguiente - valor_anterior
            ) * proporcion
            interpolados.append(tiempo_actual)

    return ResultadoInterpolacion(
        faltantes_antes=faltantes_antes,
        huecos_detectados=len(corridas),
        longitud_maxima=max(longitudes, default=0),
        valores_interpolados=len(interpolados),
        faltantes_despues=int(datos[columna].isna().sum()),
        timestamps_interpolados=tuple(interpolados),
    )


def periodo_desde_hora(hora: pd.Series) -> pd.Series:
    """Deriva periodo_dia con rangos cerrados y no solapados."""

    return pd.cut(
        hora,
        bins=[-1, 5, 11, 17, 23],
        labels=["madrugada", "mañana", "tarde", "noche"],
    ).astype("string")


def contar_cambios(anterior: pd.Series, nuevo: pd.Series) -> int:
    iguales = anterior.eq(nuevo) | (anterior.isna() & nuevo.isna())
    return int((~iguales).sum())


def lineas_conteos(conteos: dict[str, int]) -> list[str]:
    return [f"  - {columna}: {cantidad}" for columna, cantidad in conteos.items()]


def formatear_registros(registros: pd.DataFrame, columna_valor: str) -> list[str]:
    return [
        f"  - {fila.timestamp}: {columna_valor}={getattr(fila, columna_valor)}"
        for fila in registros.itertuples(index=False)
    ]


def main() -> None:
    """Ejecuta la limpieza, valida invariantes y genera el reporte."""

    if not RUTA_ENTRADA.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {RUTA_ENTRADA}")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    original = pd.read_csv(RUTA_ENTRADA, encoding="utf-8")
    filas_iniciales = len(original)

    titulo("1. EXPLORACION DEL ARCHIVO ORIGINAL")
    print(f"Archivo: {RUTA_ENTRADA}")
    print(f"Dimensiones: {original.shape[0]} filas x {original.shape[1]} columnas")
    print(f"Columnas: {original.columns.tolist()}")
    print("\nTipos de datos al cargar:")
    print(original.dtypes.to_string())
    print("\nEstadisticas descriptivas:")
    print(original.describe(include="all").T.to_string())

    if COLUMNA_TIEMPO not in original.columns:
        raise KeyError(f"Falta la columna obligatoria '{COLUMNA_TIEMPO}'.")

    duplicados_exactos = int(original.duplicated(keep="first").sum())
    limpio = original.drop_duplicates(keep="first").copy()
    limpio[COLUMNA_TIEMPO] = pd.to_datetime(limpio[COLUMNA_TIEMPO], errors="coerce")
    timestamps_invalidos = int(limpio[COLUMNA_TIEMPO].isna().sum())
    if timestamps_invalidos:
        limpio = limpio.loc[limpio[COLUMNA_TIEMPO].notna()].copy()
    limpio = limpio.sort_values(COLUMNA_TIEMPO, kind="mergesort").reset_index(drop=True)

    timestamps_duplicados = int(limpio[COLUMNA_TIEMPO].duplicated(keep=False).sum())
    if timestamps_duplicados:
        raise ValueError(
            "Hay timestamps repetidos con datos diferentes; no se elige una fila "
            "arbitrariamente sin una fuente autoritativa."
        )

    filas_tras_limpieza_base = len(limpio)
    faltantes_antes = conteos_faltantes(limpio)
    total_faltantes_antes = sum(faltantes_antes.values())

    titulo("2. INSPECCION EXACTA DE FALTANTES Y ANOMALIAS")
    print(f"Duplicados exactos eliminados: {duplicados_exactos}")
    print(f"Timestamps invalidos eliminados: {timestamps_invalidos}")
    print(f"Filas tras limpieza base: {filas_tras_limpieza_base}")
    print(f"Celdas faltantes antes del tratamiento: {total_faltantes_antes}")
    for columna, cantidad in faltantes_antes.items():
        if cantidad:
            corridas = corridas_faltantes(limpio, columna)
            maxima = max((fin - inicio + 1 for inicio, fin in corridas), default=0)
            en_extremo = bool(
                limpio[columna].isna().iloc[0] or limpio[columna].isna().iloc[-1]
            )
            print(
                f"  {columna}: {cantidad} NaN, {len(corridas)} huecos, "
                f"maximo={maxima} hora(s), en_extremo={en_extremo}"
            )

    humedad_invalida = limpio.loc[
        limpio["humedad_pct"].notna() & ~limpio["humedad_pct"].between(0, 100),
        [COLUMNA_TIEMPO, "humedad_pct"],
    ].copy()
    viento_invalido = limpio.loc[
        limpio["viento_kmh"].notna() & (limpio["viento_kmh"] < 0),
        [COLUMNA_TIEMPO, "viento_kmh"],
    ].copy()
    precios_negativos = limpio.loc[
        limpio["precio_kwh"].notna() & (limpio["precio_kwh"] < 0),
        [COLUMNA_TIEMPO, "precio_kwh"],
    ].copy()
    print(f"Humedades fuera de [0, 100]: {len(humedad_invalida)}")
    print(f"Vientos negativos: {len(viento_invalido)}")
    print(f"Precios negativos conservados: {len(precios_negativos)}")

    limpio.loc[humedad_invalida.index, "humedad_pct"] = np.nan
    limpio.loc[viento_invalido.index, "viento_kmh"] = np.nan

    titulo("3. RECONSTRUCCION DETERMINISTICA DE VARIABLES TEMPORALES")
    reconstrucciones_temporales: dict[str, int] = {}
    valores_temporales = {
        "hora": limpio[COLUMNA_TIEMPO].dt.hour.astype("int64"),
        "dia_semana": limpio[COLUMNA_TIEMPO].dt.dayofweek.astype("int64"),
        "fin_semana": (limpio[COLUMNA_TIEMPO].dt.dayofweek >= 5).astype("int64"),
        "mes": limpio[COLUMNA_TIEMPO].dt.month.astype("int64"),
    }
    for columna, valores in valores_temporales.items():
        reconstrucciones_temporales[columna] = contar_cambios(limpio[columna], valores)
        limpio[columna] = valores
        print(
            f"{columna}: {len(limpio)} reconstruidos desde timestamp; "
            f"correcciones={reconstrucciones_temporales[columna]}"
        )

    if not limpio["festivo"].dropna().isin([0, 1]).all():
        raise ValueError("festivo contiene valores distintos de 0 y 1.")
    print(
        "festivo: conservado; no es derivable sin definir un calendario oficial "
        "y una jurisdiccion."
    )

    periodo_reconstruido = periodo_desde_hora(limpio["hora"])
    periodo_original_normalizado = (
        limpio["periodo_dia"].astype("string").str.strip().str.casefold()
    )
    inconsistencias_periodo = contar_cambios(
        periodo_original_normalizado, periodo_reconstruido
    )
    cambios_textuales_periodo = contar_cambios(
        limpio["periodo_dia"], periodo_reconstruido
    )
    limpio["periodo_dia"] = periodo_reconstruido
    print(
        "periodo_dia reconstruido: 00-05=madrugada, 06-11=mañana, "
        "12-17=tarde, 18-23=noche."
    )
    print(f"Inconsistencias semanticas corregidas: {inconsistencias_periodo}")
    print(f"Valores cuyo texto cambio al reconstruir: {cambios_textuales_periodo}")

    titulo("4. INTERPOLACION TEMPORAL LOCAL")
    resultados_interpolacion: dict[str, ResultadoInterpolacion] = {}
    for columna in COLUMNAS_INTERPOLABLES:
        resultado = interpolar_huecos_locales(limpio, columna)
        resultados_interpolacion[columna] = resultado
        if resultado.faltantes_antes or resultado.valores_interpolados:
            print(
                f"{columna}: antes={resultado.faltantes_antes}, "
                f"huecos={resultado.huecos_detectados}, "
                f"maximo={resultado.longitud_maxima}, "
                f"interpolados={resultado.valores_interpolados}, "
                f"despues={resultado.faltantes_despues}"
            )
    print(
        "Regla: interpolacion lineal por timestamp solo en huecos interiores de "
        f"hasta {MAX_HORAS_INTERPOLABLES} horas con extremos validos. "
        "Sin extrapolacion ni estadisticas globales."
    )

    titulo("5. VERIFICACION DE demanda_objetivo")
    objetivo_original = limpio["demanda_objetivo"].copy()
    objetivo_esperado = limpio["demanda_mw"].shift(-1)
    comparables = objetivo_esperado.notna()
    coincidencias = np.isclose(
        objetivo_original.fillna(0),
        objetivo_esperado.fillna(0),
        rtol=0,
        atol=1e-9,
    )
    objetivos_inconsistentes = int((comparables & ~coincidencias).sum())
    limpio["demanda_objetivo"] = objetivo_esperado
    objetivos_reconstruidos = int(objetivo_esperado.notna().sum())
    print(f"Etiquetas comparables: {int(comparables.sum())}")
    print(f"Inconsistencias encontradas y corregidas: {objetivos_inconsistentes}")
    print(f"Etiquetas reconstruidas con demanda_mw.shift(-1): {objetivos_reconstruidos}")
    print("demanda_objetivo queda reservada como y y no debe incluirse en X.")

    faltantes_pre_eliminacion = limpio[COLUMNAS_REQUERIDAS_MODELO].isna()
    mascara_no_utilizable = faltantes_pre_eliminacion.any(axis=1)
    filas_eliminadas_df = limpio.loc[mascara_no_utilizable, [COLUMNA_TIEMPO]].copy()
    filas_eliminadas = int(mascara_no_utilizable.sum())
    serie_completa = limpio.copy()
    limpio = limpio.loc[~mascara_no_utilizable].reset_index(drop=True)

    demanda_por_timestamp = serie_completa.set_index(COLUMNA_TIEMPO)["demanda_mw"]
    objetivo_por_timestamp = (limpio[COLUMNA_TIEMPO] + pd.Timedelta(hours=1)).map(
        demanda_por_timestamp
    )
    objetivo_final_correcto = np.isclose(
        limpio["demanda_objetivo"], objetivo_por_timestamp, rtol=0, atol=1e-9
    )
    errores_objetivo_finales = int((~objetivo_final_correcto).sum())

    titulo("6. VALIDACIONES FINALES")
    duplicados_finales = int(limpio[COLUMNA_TIEMPO].duplicated().sum())
    orden_ascendente = bool(limpio[COLUMNA_TIEMPO].is_monotonic_increasing)
    diferencias = limpio[COLUMNA_TIEMPO].diff()
    saltos_temporales = int(
        (diferencias.notna() & diferencias.ne(pd.Timedelta(hours=1))).sum()
    )
    faltantes_despues = conteos_faltantes(limpio)
    total_faltantes_despues = sum(faltantes_despues.values())
    humedad_final_valida = bool(limpio["humedad_pct"].between(0, 100).all())
    viento_final_valido = bool((limpio["viento_kmh"] >= 0).all())

    print(f"Timestamp ordenado: {orden_ascendente}")
    print(f"Timestamps duplicados: {duplicados_finales}")
    print(f"Saltos horarios: {saltos_temporales}")
    print(f"NaN en columnas requeridas: {total_faltantes_despues}")
    print(f"Humedad dentro de [0,100]: {humedad_final_valida}")
    print(f"Viento >= 0: {viento_final_valido}")
    print(f"Errores finales en demanda_objetivo: {errores_objetivo_finales}")

    if not orden_ascendente or duplicados_finales:
        raise AssertionError("Fallo la validacion temporal final.")
    if total_faltantes_despues:
        raise AssertionError("Persisten NaN en columnas requeridas.")
    if not humedad_final_valida or not viento_final_valido:
        raise AssertionError("Persisten valores fisicamente invalidos.")
    if errores_objetivo_finales:
        raise AssertionError("demanda_objetivo no coincide con timestamp + 1 hora.")

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    limpio.to_csv(
        RUTA_SALIDA,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    rango_temporal = (
        f"{limpio[COLUMNA_TIEMPO].min()} a {limpio[COLUMNA_TIEMPO].max()}"
        if not limpio.empty
        else "sin registros"
    )

    reporte: list[str] = [
        "REPORTE DE LIMPIEZA DEFINITIVA",
        "=" * 80,
        "",
        "1. FILAS Y ESTRUCTURA",
        f"Filas iniciales del CSV original: {filas_iniciales}",
        f"Duplicados encontrados/eliminados: {duplicados_exactos}/{duplicados_exactos}",
        f"Timestamps invalidos encontrados/eliminados: {timestamps_invalidos}/{timestamps_invalidos}",
        f"Filas tras limpieza base: {filas_tras_limpieza_base}",
        f"Filas eliminadas en el cierre: {filas_eliminadas}",
        f"Filas finales: {len(limpio)}",
        "",
        "2. VALORES FALTANTES POR COLUMNA ANTES",
        *lineas_conteos(faltantes_antes),
        f"Total antes: {total_faltantes_antes}",
        "",
        "3. VALORES CORREGIDOS Y RECONSTRUIDOS",
        (
            "periodo_dia: 00-05=madrugada, 06-11=mañana, "
            "12-17=tarde, 18-23=noche."
        ),
        (
            f"periodo_dia reconstruidos={filas_tras_limpieza_base}; "
            f"inconsistencias_semanticas_corregidas={inconsistencias_periodo}; "
            f"textos_modificados={cambios_textuales_periodo}"
        ),
        *[
            f"{columna}: reconstruidos={filas_tras_limpieza_base}; corregidos={cantidad}"
            for columna, cantidad in reconstrucciones_temporales.items()
        ],
        (
            "festivo conservado: no es derivable sin calendario oficial y jurisdiccion."
        ),
        f"humedad_pct fuera de [0,100] convertida a NaN: {len(humedad_invalida)}",
        *formatear_registros(humedad_invalida, "humedad_pct"),
        f"viento_kmh negativo convertido a NaN: {len(viento_invalido)}",
        *formatear_registros(viento_invalido, "viento_kmh"),
        "",
        "4. INTERPOLACIONES LOCALES",
        (
            "Interpolacion lineal por timestamp solo para huecos interiores de hasta "
            f"{MAX_HORAS_INTERPOLABLES} horas con extremos validos. Sin extrapolacion "
            "ni estadisticas globales."
        ),
        *[
            (
                f"{columna}: antes={resultado.faltantes_antes}, "
                f"huecos={resultado.huecos_detectados}, "
                f"longitud_maxima={resultado.longitud_maxima}, "
                f"interpolados={resultado.valores_interpolados}, "
                f"despues={resultado.faltantes_despues}"
            )
            for columna, resultado in resultados_interpolacion.items()
            if resultado.faltantes_antes or resultado.valores_interpolados
        ],
        "",
        "5. FILAS ELIMINADAS",
        (
            "Solo se eliminan filas que aun tienen NaN requerido. La ultima hora no "
            "puede etiquetarse porque no existe demanda de la hora siguiente."
        ),
        *(
            [f"  - {fila.timestamp}" for fila in filas_eliminadas_df.itertuples(index=False)]
            if filas_eliminadas
            else ["  - Ninguna"]
        ),
        "",
        "6. ANOMALIAS CONSERVADAS Y JUSTIFICACION",
        (
            f"Precios negativos conservados: {len(precios_negativos)}. Pueden ser "
            "validos en mercados electricos y no hay evidencia interna de corrupcion."
        ),
        *formatear_registros(precios_negativos, "precio_kwh"),
        (
            "Otros extremos se conservaron: un extremo estadistico sin regla fisica "
            "o evidencia externa no demuestra que el dato sea erroneo."
        ),
        "",
        "7. VERIFICACION DE demanda_objetivo",
        "Regla: demanda_objetivo(t) = demanda_mw(t + 1 hora).",
        f"Inconsistencias detectadas y corregidas: {objetivos_inconsistentes}",
        f"Etiquetas reconstruidas: {objetivos_reconstruidos}",
        f"Errores finales por timestamp + 1 hora: {errores_objetivo_finales}",
        "demanda_objetivo se reserva como y y no debe incluirse en X.",
        "",
        "8. VALORES FALTANTES POR COLUMNA DESPUES",
        *lineas_conteos(faltantes_despues),
        f"Total despues: {total_faltantes_despues}",
        "",
        "9. VALIDACION FINAL",
        f"Rango temporal: {rango_temporal}",
        f"Timestamp ordenado ascendentemente: {orden_ascendente}",
        f"Timestamps duplicados: {duplicados_finales}",
        f"Saltos temporales distintos de una hora: {saltos_temporales}",
        f"Humedad dentro de [0,100]: {humedad_final_valida}",
        f"Viento mayor o igual a 0: {viento_final_valido}",
        "",
        "No se aplicaron normalizacion, division train/validation/test, ventanas,",
        "modelos LSTM ni entrenamiento.",
    ]
    RUTA_REPORTE.parent.mkdir(parents=True, exist_ok=True)
    RUTA_REPORTE.write_text("\n".join(reporte) + "\n", encoding="utf-8")

    titulo("7. RESUMEN FINAL")
    print(f"Filas iniciales: {filas_iniciales}")
    print(f"Filas tras eliminar duplicados: {filas_tras_limpieza_base}")
    print(f"Filas finales: {len(limpio)}")
    print(f"Filas eliminadas en el cierre: {filas_eliminadas}")
    print(f"Faltantes antes/despues: {total_faltantes_antes}/{total_faltantes_despues}")
    print(f"periodo_dia inconsistentes corregidos: {inconsistencias_periodo}")
    print(f"periodo_dia textos modificados: {cambios_textuales_periodo}")
    print(
        "Interpolados: "
        + ", ".join(
            f"{columna}={resultado.valores_interpolados}"
            for columna, resultado in resultados_interpolacion.items()
            if resultado.valores_interpolados
        )
    )
    print(f"demanda_objetivo corregidos: {objetivos_inconsistentes}")
    print(f"Precios negativos conservados: {len(precios_negativos)}")
    print(f"Rango temporal final: {rango_temporal}")
    print(f"Saltos temporales: {saltos_temporales}")
    print(f"Dataset definitivo: {RUTA_SALIDA}")
    print(f"Reporte: {RUTA_REPORTE}")
    print("No se realizo ninguna etapa de modelado.")


if __name__ == "__main__":
    main()
