# P2
# **Proyecto P2 - Clasificación en la Competencia "Spaceship Titanic"**

## **Introducción**
Este proyecto tiene como objetivo desarrollar modelos de clasificación para predecir el estado de los pasajeros en la competencia de Kaggle **"Spaceship Titanic"**. A través de diversas técnicas de aprendizaje supervisado, buscamos identificar el mejor modelo para generar predicciones precisas y competitivas.

## **Objetivo**
El propósito de este análisis es explorar los datos proporcionados, entrenar y evaluar distintos modelos de clasificación, y seleccionar el más adecuado con base en su desempeño en un conjunto de prueba. Finalmente, se generarán predicciones que serán enviadas a Kaggle para medir la precisión del modelo en la competencia.

## **Metodología**
Para lograr este objetivo, seguiremos los siguientes pasos:
1. **Exploración de los datos:** Identificación de variables cualitativas y cuantitativas, detección de valores faltantes y análisis preliminar.
2. **Entrenamiento de modelos:** Implementación de diversas técnicas de clasificación, incluyendo:
   - Regresión Logística Multinomial
   - Análisis Discriminante Lineal (LDA)
   - Árbol de Decisión
   - Métodos de Ensemble (Random Forest y Boosting)
3. **Evaluación con validación cruzada:** Comparación de los modelos con la métrica de *accuracy* para determinar su rendimiento.
4. **Selección del mejor modelo:** Se comparará el desempeño en un conjunto de prueba (`X_test`) para elegir el modelo con mejor precisión.
5. **Generación de predicciones:** Se aplicará el modelo seleccionado a los datos de prueba y se generará un archivo de predicciones para Kaggle.

## **Criterio de Evaluación**
La calidad del modelo será medida mediante la métrica de **accuracy**, tanto en validación cruzada como en datos de prueba. El objetivo es obtener un puntaje competitivo en la plataforma Kaggle y documentar el proceso de manera clara y reproducible.

Este documento servirá como una guía detallada del desarrollo del proyecto, explicando cada paso de la implementación y las decisiones tomadas a lo largo del análisis.

Este proyecto incluye los siguientes documentos:
- [Reporte en formato ipynb](./P2%20504065.ipynb)
- [Reporte en formato pdf](./P2%20504065.pdf)
- [Reporte en formato html](./P2%20504065.html)
- [Conunto de datos para Training](./train.csv)
- [Conjunto de datos para Test](./test.csv)
