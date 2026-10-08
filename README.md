# Despliegue https://ia-taller-ltsm-bcnaq27gdo6gls3s5aqkux.streamlit.app/

# Predicción de Demanda Energética con LSTM

Proyecto de predicción de demanda energética mediante redes LSTM. Incluye entrenamiento, evaluación, gráficas y una aplicación web desarrollada con Streamlit.

La metodología, las arquitecturas y el análisis de resultados se encuentran explicados en el documento del taller.

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

Crear y activar el entorno virtual (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

## Ejecutar aplicación web

```powershell
streamlit run app.py
```

Abrir en el navegador:

http://localhost:8501

La aplicación permite ingresar variables independientes (X) y obtener la predicción de demanda energética (Y) en MW utilizando el modelo LSTM entrenado.

## Visualizar las gráficas

Las gráficas generadas se encuentran en:

`results/graficas/`

Para abrir la carpeta desde PowerShell:

```powershell
ii .\results\graficas
```

Las cuatro visualizaciones principales solicitadas son:

1. Loss de entrenamiento vs. validación.
2. Demanda real vs. predicha.
3. Error de predicción.
4. Comparación de modelos.

Se encuentran disponibles 28 gráficas correspondientes a los nueve experimentos y su comparación general.

Para regenerarlas a partir de los resultados guardados:

```powershell
python -m src.graficas
```

## Consultar resultados

La tabla con las métricas de los nueve experimentos está disponible en:

`results/resultados_modelos.csv`

Para abrirla en Windows:

```powershell
ii .\results\resultados_modelos.csv
```

## Comandos adicionales

Ejecutar limpieza:

```powershell
python -m src.limpieza
```

Preparar los datos:

```powershell
python -m src.preparacion
```

Entrenar los modelos:

```powershell
python -m src.entrenamiento
```

Evaluar los modelos guardados:

```powershell
python -m src.evaluacion
```

**Nota:** no es necesario volver a entrenar para utilizar la aplicación o consultar los resultados existentes. Para reproducir los experimentos desde cero, ejecutar los procesos en el orden indicado. El entrenamiento puede tardar varios minutos.
