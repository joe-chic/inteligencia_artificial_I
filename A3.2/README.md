# A3.2 Redes Neuronales

Este proyecto implementa un clasificador de dígitos manuscritos basado en redes neuronales profundas, usando el conjunto de datos MNIST. A lo largo del desarrollo se abordan todas las etapas clásicas de un flujo de trabajo en visión por computadora: carga y preprocesamiento de imágenes, diseño y entrenamiento de modelos, validación, evaluación y despliegue de un prototipo de reconocimiento en tiempo real mediante cámara.

## Objetivo

Desarrollar y comparar diferentes arquitecturas de redes neuronales (densas y convolucionales) que logren alta precisión en la clasificación de dígitos manuscritos, garantizando al mismo tiempo una buena capacidad de generalización. Se busca además diagnosticar y mitigar fenómenos de sobreajuste, así como validar el comportamiento del modelo en varios subconjuntos de datos (entrenamiento, validación, prueba y muestras seleccionadas).

## Propósito

Ofrecer un ejemplo práctico e integral de los conceptos vistos en clase sobre redes neuronales:  
- **Funciones de activación** (ReLU, softmax)  
- **Funciones de pérdida** (entropía cruzada categórica)  
- **Optimizadores adaptativos** (Adam)  
- **Regularización** (L2, Dropout)  
- **Data augmentation** (rotaciones, desplazamientos, zoom, ajuste de brillo)  

Además, brindar una base extensible para explorar mejoras en preprocesamiento, arquitecturas y técnicas avanzadas de augmentación, y medir su impacto sobre la precisión y robustez del sistema tanto en pruebas estáticas como en tiempo real.  
