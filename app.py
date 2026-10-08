"""Interfaz Streamlit para predecir la demanda de la siguiente hora.

La aplicacion usa el modelo base con ventana de 48 horas y exclusivamente los
transformadores ya ajustados durante la preparacion. No reentrena ni reajusta
ningun artefacto.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

from src.modelos import crear_modelo
from src.preparacion import (
    FEATURES_NUMERICAS,
    FEATURES_ORIGINALES,
    PERIODOS_ORDENADOS,
)


RAIZ_PROYECTO = Path(__file__).resolve().parent
RUTA_DATASET = RAIZ_PROYECTO / "data" / "dataset_limpio.csv"
RUTA_MODELO = RAIZ_PROYECTO / "models" / "base_v48.keras"
RUTA_SCALER_X = RAIZ_PROYECTO / "models" / "scaler_X.pkl"
RUTA_SCALER_Y = RAIZ_PROYECTO / "models" / "scaler_y.pkl"
RUTA_ENCODER = RAIZ_PROYECTO / "models" / "encoder_periodo_dia.pkl"
RUTA_FEATURES = RAIZ_PROYECTO / "models" / "features.txt"

COLUMNA_TIEMPO = "timestamp"
TARGET = "demanda_objetivo"
VENTANA = 48
COLUMNAS_ENTRADA = [COLUMNA_TIEMPO, *FEATURES_ORIGINALES]


def exigir(condicion: bool, mensaje: str) -> None:
    """Genera un error de validacion legible para la interfaz."""

    if not condicion:
        raise ValueError(mensaje)


@st.cache_resource(show_spinner="Cargando modelo y preprocesadores...")
def cargar_artefactos() -> tuple[keras.Model, object, object, object, list[str]]:
    """Carga artefactos ya ajustados sin ejecutar fit ni compile."""

    rutas = [
        RUTA_MODELO,
        RUTA_SCALER_X,
        RUTA_SCALER_Y,
        RUTA_ENCODER,
        RUTA_FEATURES,
    ]
    faltantes = [str(ruta) for ruta in rutas if not ruta.exists()]
    exigir(not faltantes, f"Faltan artefactos requeridos: {faltantes}")

    scaler_x = joblib.load(RUTA_SCALER_X)
    scaler_y = joblib.load(RUTA_SCALER_Y)
    encoder = joblib.load(RUTA_ENCODER)
    features_finales = [
        linea.strip()
        for linea in RUTA_FEATURES.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]

    # El archivo fue generado con una version reciente de Keras. Reconstruir la
    # arquitectura conocida y cargar sus pesos evita depender de detalles de
    # serializacion entre versiones, sin modificar ni reentrenar el modelo.
    modelo = crear_modelo("base", (VENTANA, len(features_finales)))
    modelo.optimizer = None
    modelo.load_weights(RUTA_MODELO)

    exigir(hasattr(scaler_x, "transform"), "scaler_X.pkl no es valido.")
    exigir(hasattr(scaler_y, "inverse_transform"), "scaler_y.pkl no es valido.")
    exigir(hasattr(encoder, "transform"), "encoder_periodo_dia.pkl no es valido.")
    exigir(TARGET not in features_finales, "demanda_objetivo aparece en features.txt.")
    exigir(COLUMNA_TIEMPO not in features_finales, "timestamp aparece en features.txt.")
    exigir(
        tuple(modelo.input_shape[1:]) == (VENTANA, len(features_finales)),
        (
            "El modelo no coincide con la ventana o las features guardadas: "
            f"modelo={modelo.input_shape}, esperada=({VENTANA}, {len(features_finales)})."
        ),
    )
    exigir(modelo.output_shape[-1] == 1, "El modelo no produce una salida escalar.")
    return modelo, scaler_x, scaler_y, encoder, features_finales


@st.cache_data(show_spinner=False)
def cargar_ejemplo() -> pd.DataFrame:
    """Carga las últimas 48 horas limpias sin incluir el target."""

    exigir(RUTA_DATASET.exists(), f"No existe {RUTA_DATASET}.")
    datos = pd.read_csv(RUTA_DATASET, encoding="utf-8")
    faltantes = sorted(set(COLUMNAS_ENTRADA) - set(datos.columns))
    exigir(not faltantes, f"El dataset limpio no contiene: {faltantes}.")
    exigir(len(datos) >= VENTANA, "El dataset limpio no contiene 48 horas.")
    return datos[COLUMNAS_ENTRADA].tail(VENTANA).reset_index(drop=True)


def periodo_esperado_desde_hora(hora: pd.Series) -> pd.Series:
    """Aplica la misma regla temporal utilizada durante la limpieza."""

    return pd.cut(
        hora,
        bins=[-1, 5, 11, 17, 23],
        labels=PERIODOS_ORDENADOS,
    ).astype("string")


def validar_entrada(datos: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Valida esquema, tipos y continuidad sin modificar el orden recibido."""

    exigir(isinstance(datos, pd.DataFrame), "La entrada no es una tabla valida.")
    entrada = datos.copy()
    columnas_ignoradas: list[str] = []

    if TARGET in entrada.columns:
        columnas_ignoradas.append(TARGET)
        entrada = entrada.drop(columns=[TARGET])

    faltantes = sorted(set(COLUMNAS_ENTRADA) - set(entrada.columns))
    exigir(not faltantes, f"Faltan columnas obligatorias: {faltantes}.")
    extras = sorted(set(entrada.columns) - set(COLUMNAS_ENTRADA))
    columnas_ignoradas.extend(extras)
    entrada = entrada[COLUMNAS_ENTRADA].copy()

    exigir(
        len(entrada) == VENTANA,
        f"Se requieren exactamente {VENTANA} filas; se recibieron {len(entrada)}.",
    )
    entrada[COLUMNA_TIEMPO] = pd.to_datetime(
        entrada[COLUMNA_TIEMPO], errors="coerce"
    )
    exigir(
        entrada[COLUMNA_TIEMPO].notna().all(),
        "timestamp contiene valores vacios o invalidos.",
    )
    exigir(
        entrada[COLUMNA_TIEMPO].is_monotonic_increasing,
        "Los timestamps deben venir ordenados ascendentemente.",
    )
    exigir(
        not entrada[COLUMNA_TIEMPO].duplicated().any(),
        "Los timestamps deben ser unicos.",
    )
    diferencias = entrada[COLUMNA_TIEMPO].diff().dropna()
    exigir(
        diferencias.eq(pd.Timedelta(hours=1)).all(),
        "Las 48 filas deben ser horas consecutivas, sin saltos.",
    )

    for columna in FEATURES_NUMERICAS:
        entrada[columna] = pd.to_numeric(entrada[columna], errors="coerce")
    exigir(
        entrada[FEATURES_NUMERICAS].notna().all().all(),
        "Las variables numericas contienen vacios o texto no numerico.",
    )
    exigir(
        np.isfinite(entrada[FEATURES_NUMERICAS].to_numpy(dtype=np.float64)).all(),
        "Las variables numericas contienen infinitos.",
    )

    periodos = entrada["periodo_dia"].astype("string")
    exigir(periodos.notna().all(), "periodo_dia contiene valores vacios.")
    categorias_invalidas = sorted(set(periodos.astype(str)) - set(PERIODOS_ORDENADOS))
    exigir(
        not categorias_invalidas,
        f"periodo_dia contiene categorias invalidas: {categorias_invalidas}.",
    )
    entrada["periodo_dia"] = periodos

    tiempo = entrada[COLUMNA_TIEMPO]
    exigir(
        entrada["hora"].eq(tiempo.dt.hour).all(),
        "hora no coincide con timestamp.",
    )
    exigir(
        entrada["dia_semana"].eq(tiempo.dt.dayofweek).all(),
        "dia_semana no coincide con timestamp (lunes=0, domingo=6).",
    )
    exigir(
        entrada["fin_semana"].eq((tiempo.dt.dayofweek >= 5).astype(int)).all(),
        "fin_semana no coincide con timestamp.",
    )
    exigir(
        entrada["mes"].eq(tiempo.dt.month).all(),
        "mes no coincide con timestamp.",
    )
    exigir(
        entrada["periodo_dia"].eq(periodo_esperado_desde_hora(entrada["hora"])).all(),
        (
            "periodo_dia no coincide con hora. Regla: 00-05 madrugada, "
            "06-11 mañana, 12-17 tarde y 18-23 noche."
        ),
    )
    exigir(entrada["festivo"].isin([0, 1]).all(), "festivo debe contener solo 0 o 1.")
    exigir(
        entrada["humedad_pct"].between(0, 100).all(),
        "humedad_pct debe estar entre 0 y 100.",
    )
    exigir((entrada["viento_kmh"] >= 0).all(), "viento_kmh no puede ser negativo.")
    return entrada, sorted(set(columnas_ignoradas))


def preprocesar(
    entrada: pd.DataFrame,
    scaler_x: object,
    encoder: object,
    features_finales_guardadas: list[str],
) -> np.ndarray:
    """Replica el encoding y escalado de src/preparacion.py usando transform."""

    numericas = entrada[FEATURES_NUMERICAS].to_numpy(dtype=np.float64)
    periodo_codificado = encoder.transform(entrada[["periodo_dia"]])
    if hasattr(periodo_codificado, "toarray"):
        periodo_codificado = periodo_codificado.toarray()
    periodo_codificado = np.asarray(periodo_codificado, dtype=np.float64)

    nombres_one_hot = encoder.get_feature_names_out(["periodo_dia"]).tolist()
    features_calculadas = [*FEATURES_NUMERICAS, *nombres_one_hot]
    exigir(
        features_calculadas == features_finales_guardadas,
        (
            "El orden de features producido por el encoder no coincide con "
            "models/features.txt."
        ),
    )

    matriz = np.concatenate([numericas, periodo_codificado], axis=1)
    exigir(
        matriz.shape == (VENTANA, len(features_finales_guardadas)),
        f"La matriz de entrada tiene una forma inesperada: {matriz.shape}.",
    )
    matriz_escalada = scaler_x.transform(matriz)
    exigir(
        np.isfinite(matriz_escalada).all(),
        "El preprocesamiento produjo NaN o infinitos.",
    )
    return np.asarray(matriz_escalada, dtype=np.float32).reshape(
        1, VENTANA, len(features_finales_guardadas)
    )


def predecir_demanda(
    entrada: pd.DataFrame,
    modelo: keras.Model,
    scaler_x: object,
    scaler_y: object,
    encoder: object,
    features_finales: list[str],
) -> float:
    """Predice y devuelve un único valor en MW."""

    x_modelo = preprocesar(entrada, scaler_x, encoder, features_finales)
    prediccion_escalada = modelo.predict(x_modelo, verbose=0)
    exigir(
        prediccion_escalada.shape == (1, 1),
        f"El modelo produjo una forma inesperada: {prediccion_escalada.shape}.",
    )
    prediccion_mw = float(scaler_y.inverse_transform(prediccion_escalada)[0, 0])
    exigir(np.isfinite(prediccion_mw), "La prediccion no es un numero finito.")
    return prediccion_mw


def main() -> None:
    """Construye la interfaz y ejecuta una prediccion solo al pulsar el boton."""

    st.set_page_config(
        page_title="Prediccion de demanda energetica",
        page_icon="⚡",
        layout="wide",
    )
    st.title("Predicción de demanda energética")
    st.caption(
        "Modelo LSTM base · ventana de 48 horas · predicción de la siguiente hora"
    )

    try:
        modelo, scaler_x, scaler_y, encoder, features_finales = cargar_artefactos()
        ejemplo = cargar_ejemplo()
    except Exception as error:
        st.error(f"No fue posible iniciar la aplicación: {error}")
        st.stop()

    modo = st.radio(
        "Fuente de las 48 horas históricas",
        options=["Editar ejemplo precargado", "Cargar archivo CSV"],
        horizontal=True,
    )

    if modo == "Editar ejemplo precargado":
        st.info(
            "La tabla contiene las últimas 48 horas de data/dataset_limpio.csv. "
            "Puedes editar sus valores antes de predecir."
        )
        datos_iniciales = ejemplo
    else:
        archivo = st.file_uploader(
            "Selecciona un CSV con exactamente 48 horas consecutivas",
            type=["csv"],
        )
        st.download_button(
            "Descargar CSV de ejemplo",
            data=ejemplo.to_csv(index=False).encode("utf-8"),
            file_name="ejemplo_48_horas.csv",
            mime="text/csv",
        )
        if archivo is None:
            st.info("Carga un archivo CSV para continuar.")
            st.stop()
        try:
            datos_iniciales = pd.read_csv(archivo, encoding="utf-8")
        except Exception as error:
            st.error(f"No fue posible leer el CSV: {error}")
            st.stop()

    st.subheader("Datos históricos")
    datos_editados = st.data_editor(
        datos_iniciales,
        num_rows="fixed",
        hide_index=True,
        width="stretch",
        height=520,
    )
    st.caption(
        "`timestamp` y `demanda_objetivo` nunca se usan como features. "
        "Los artefactos existentes se aplican únicamente con transform()."
    )

    if st.button("Predecir demanda de la siguiente hora", type="primary"):
        try:
            entrada_validada, columnas_ignoradas = validar_entrada(datos_editados)
            if columnas_ignoradas:
                st.warning(
                    "Columnas ignoradas y excluidas de la entrada del modelo: "
                    + ", ".join(columnas_ignoradas)
                )
            with st.spinner("Calculando predicción..."):
                demanda_mw = predecir_demanda(
                    entrada_validada,
                    modelo,
                    scaler_x,
                    scaler_y,
                    encoder,
                    features_finales,
                )
            siguiente_hora = (
                entrada_validada[COLUMNA_TIEMPO].iloc[-1] + pd.Timedelta(hours=1)
            )
            st.success("Predicción calculada correctamente.")
            columna_metrica, columna_tiempo = st.columns(2)
            columna_metrica.metric(
                "Demanda predicha",
                f"{demanda_mw:,.2f} MW",
            )
            columna_tiempo.metric(
                "Hora objetivo",
                siguiente_hora.strftime("%Y-%m-%d %H:%M"),
            )
        except Exception as error:
            st.error(f"No se pudo realizar la predicción: {error}")


if __name__ == "__main__":
    main()
