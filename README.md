# BrainTumorDetection-YOLOv11-MRI

El objetivo es detectar tumores cerebrales en **resonancias magnéticas (MRI)** utilizando **YOLOv11**. Las resonancias magnéticas son inicialmente imágenes 3D, que luego se procesan para obtener imágenes 2D, desde el plano axial, para su análisis. El modelo ha sido entrenado para predecir la presencia de tumores en función de estas imágenes procesadas.

## Procesamiento de Datos

1. **Carga de Imágenes 3D de MRI**  
   El conjunto de datos consta de resonancias magnéticas 3D del cerebro. Estas imágenes 3D se cargan y luego se procesan para generar **cortes 2D** de las resonancias. Los cortes se toman desde el plano axial

2. **Extracción del Ground Truth**  
   Las etiquetas de **ground truth** se extraen de las imágenes de MRI, que representan las ubicaciones exactas de los tumores en el cerebro. Esta información es crucial para entrenar el modelo y evaluar su rendimiento.

3. **Conversión al Formato YOLOv11**  
   Las anotaciones de ground truth se convierten luego al **formato YOLOv11**, que es el formato requerido para entrenar el modelo YOLO. En este formato:
   - Cada imagen tiene un archivo `.txt` asociado con las anotaciones.
   - Cada línea en el archivo `.txt` corresponde a un cuadro delimitador alrededor del tumor.
   - El formato de anotación para cada cuadro delimitador es el siguiente:
     ```
     class_id center_x center_y width height
     ```
     - `class_id`: La etiqueta de clase (en este caso, 0 para un tumor cerebral).
     - `center_x`, `center_y`: El centro del cuadro delimitador en coordenadas normalizadas (valores entre 0 y 1).
     - `width`, `height`: El ancho y la altura del cuadro delimitador, también en coordenadas normalizadas.

4. **Preparación de Imágenes para YOLOv11**  
   Después de convertir las anotaciones de ground truth al formato YOLOv11, los cortes de MRI están listos para ser utilizados en el entrenamiento del modelo.

## Entrenamiento del Modelo

El modelo fue entrenado utilizando la arquitectura **YOLOv11**. El proceso de entrenamiento incluyó los siguientes parámetros:

- **Patience**: El entrenamiento se configuró con una **paciencia de 7 épocas**, lo que significa que si el rendimiento del modelo no mejoraba durante 7 épocas consecutivas, el entrenamiento se detendría antes para evitar sobreajuste.
  
- **Convergencia**: El proceso de entrenamiento continuó hasta que el modelo alcanzó la convergencia, es decir, cuando logró un rendimiento óptimo en el conjunto de validación. El entrenamiento se detuvo una vez que la pérdida dejó de mejorar significativamente.

## Evaluación del Modelo y Predicciones

Después de entrenar, el modelo fue evaluado en un conjunto de validación separado de imágenes de RMN. Los resultados fueron utilizados para evaluar la precisión y la fiabilidad de la detección de tumores.

### Ground Truth vs Predicciones

A continuación, se muestran dos imágenes para la comparación: una mostrando las **etiquetas de ground truth** (las ubicaciones reales de los tumores) y la otra mostrando las **predicciones** realizadas por el modelo. La comparación ayuda a visualizar la precisión de las predicciones del modelo.

- **Ground Truth**: Las ubicaciones reales de los tumores anotadas en las resonancias magnéticas. (marcados en color naranja)
- **Predicciones**: Las ubicaciones de los tumores predichas por el modelo YOLOv11 entrenado. (marcados con un recuadro verde)

![image](https://github.com/user-attachments/assets/766bc2d5-e5ef-4f35-a412-a7a497cee208)
Strokecase0003

![image](https://github.com/user-attachments/assets/74837ad5-1d48-42cb-80d3-c7eeae30279c)
Strokecase0013


