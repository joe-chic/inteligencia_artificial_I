# Análisis Exploratorio de Datos Básico
En este proyecto se pretende explorar de forma básica los niveles de obesidad de la población latinoamericana. Para lograrlo, se utilizó una base de datos creada por científicos de la Universidad de la Costa en Colombia, que contiene información de individuos de Colombia, Perú, y México. La base de datos original se encuentra en el [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition), pero en este proyecto se trabajó con una versión simplificada, que se encuentra en este proyecto con el nombre "A1.1 Obesidad.csv".

La base de datos cuenta con la siguiente información:
-	“Sexo”. Se describe como femenino (Female) o masculino (Male).
-	“Edad”. Se describe como un número entre 14 y 61.
-	“Estatura”. Se describe como un número, en metros.
-	“Peso”. Se describe como un número, en kilogramos.
-	“FamiliarConSobrepeso”. Describe si algún familiar ha sufrido sobrepeso (yes) o no (no).
-	“ComeMuchasCalorias”. Describe si el individuo come comidas con alto contenido calórico de forma frecuente (yes) o no (no)
-	“ComeVegetales”. Indica si el individuo nunca come vegetales en sus comidas (1), si lo hace algunas veces (2), o si lo hace siempre (3)
-	“Fumador”. Indica si la persona es fumadora activa (yes) o no (no)
-	“ConsumoDeAgua”. Indica si la persona toma menos de un litro de agua al día (1), entre uno y dos litros de agua al día (2), o más de dos litros de agua al día (3)
-	“NivelDeObesidad”. Se calcula a partir del índice de masa corporal (peso dividido entre estatura al cuadrado), y se categoriza como: bajo peso (Insufficient_Weight) para valores menores a 18.5, peso normal (Normal_Weight) para valores entre 18.5 y 24.9, sobrepeso tipo I (Overweight_Level_I) y sobrepeso tipo II (Overweight_Level_II) para valores entre 25.0 y 29.9, obesidad tipo I (Obesity_Type_I) para valores entre 30.00 y 34.9, obesidad tipo II (Obesity_Type_II) para valores entre 35.0 y 39.9, y obesidad tipo III (Obesity_Type_III) para valores superiores a 40.0

Este proyecto incluye los siguientes documentos:
- [Reporte en formato ipynb](./prueba.ipynb)
- [Reporte en formato html](./prueba.html)
- [Base de datos](./datos.csv)