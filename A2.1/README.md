# A2.1
El siguiente trabajo tiene como objetivo predecir si la calidad del aire en Monterrey, Nuevo León, es dañina para la salud, considerando distintos factores ambientales y meteorológicos. Para ello, se utilizará una base de datos con información sobre la concentración de contaminantes en el aire, condiciones atmosféricas y variaciones temporales, obtenida de la Secretaría del Medio Ambiente del Estado de Nuevo León.

La variable de salida será PM2.5, la cual se binarizará para indicar si la calidad del aire es perjudicial o no. Se considera dañina cuando los valores de PM2.5 son iguales o superiores a 35.5 µg/m³, umbral en el que grupos sensibles pueden experimentar efectos adversos.

Para el análisis, se empleará un modelo de regresión logística, evaluado mediante validación cruzada, y se determinará la influencia de cada variable en la clasificación de la calidad del aire. Además, se graficará la curva ROC y se calculará el AUC para medir el desempeño del modelo en la discriminación de las categorías establecidas.

Este proyecto incluye los siguientes documentos:
- [Reporte en formato ipynb](./A2.1%20504065.ipynb)
- [Reporte en formato pdf](./A2.1%20504065.pdf)
- [Reporte en formato html](./A2.1%20504065.html)