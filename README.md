
# DermaScan
Enlaces a colabs donde esta todo mas detallado.
- Modelo de sentimientos: https://github.com/juanjo-18/DermaScan/blob/main/colabs/modelo_sentimientos.ipynb
- Modelo de objeto y piel:
- Modelo de piel sana y piel con lesion:
- Modelo de benigno y maligno:

# Índice

1. [Justificación y descripción del proyecto](#Justificación-y-descripción-del-proyecto)
2. [Obtención de los datos](#2-obtención-de-los-datos)
3. [Limpieza de datos (Preprocesado)](#3-limpieza-de-datos-preprocesado)
4. [Exploración y visualización de los datos](#4-exploración-y-visualización-de-los-datos)
5. [Preparación de los datos para Machine Learning](#5-preparación-de-los-datos-para-machine-learning)
6. [Entrenamiento del modelo y comprobación del rendimiento](#6-entrenamiento-del-modelo-y-comprobación-del-rendimiento)
7. [Procesamiento de Lenguaje Natural](#7-procesamiento-de-lenguaje-natural)
8. [Aplicación web](#8-aplicación-web)
9. [Conclusiones](#9-conclusiones)




   
## 1. Justificación y descripción del proyecto
Nuestro innovador proyecto es una plataforma web dedicada a la detección de tumores de piel a través de la carga de imágenes. En este espacio interactivo, los usuarios podrán cargar sus imágenes y se aplicarn una variedad de filtros especializados. Desde distinguir entre objetos y áreas de la piel hasta analizar la salud cutánea y detectar posibles tumores, nuestro sistema ofrece una experiencia completa.

Con herramientas avanzadas, los usuarios podrán obtener resultados precisos sobre la naturaleza del tejido cutáneo en la imagen, identificando si es sano o presenta algún tipo de tumor. Además, ofrecemos la capacidad de diferenciar entre tumores benignos y malignos, brindando información crucial para la toma de decisiones médicas.

Este servicio va más allá al proporcionar detalles específicos sobre el tipo de tumor detectado, permitiendo a los usuarios obtener información detallada sobre su condición. Nuestra misión es hacer que la detección temprana de problemas de piel sea accesible y efectiva, brindando a los usuarios la tranquilidad y la información necesaria para tomar decisiones informadas sobre su salud cutánea. Bienvenido a una nueva era de cuidado personalizado y empoderamiento a través de la tecnología."
## 2. Obtención de los datos
## 3. Limpieza de datos (Preprocesado)


Para el modelo de sentimientos primero los twits que tenemos les estamos eliminadon valores y caracteres que no son necesarios, como los iconos, eliminar los @ y el texto asociado, eliminar # y su texto, eliminar urls y convertir todo a minusculas.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/Limpiar%20textos.png)

Aqui estamos eliminado de las frases las stopwords para despues pasarselo al modelo.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/quitamos_stopword.png)

## 4. Exploración y visualización de los datos
## 5. Preparación de los datos para Machine Learning
## 6. Entrenamiento del modelo y comprobación del rendimiento
## 7. Procesamiento de Lenguaje Natural


En el cuaderno jupyter a continuacion esta todo mas detallado del modelo.
- Modelo de sentimientos: https://github.com/juanjo-18/DermaScan/blob/main/colabs/modelo_sentimientos.ipynb

En el ámbito del Procesamiento de Lenguaje Natural, hemos desarrollado un modelo diseñado para detectar sentimientos en texto. Este modelo se integra en nuestra plataforma web, específicamente en un sistema de reseñas con calificación estelar. Al introducir un texto en el modelo, obtendrás una puntuación en una escala del 0 al 5. Una puntuación de 0 indica una valoración muy negativa, 2.5 representa una valoración neutra, y 5 refleja una valoración muy positiva.

Para conseguir crear el modelo primero necesitamos datos de textos con su puntuacion lo que hemos echo es recopilar texto de twitter y crear una serie de funciones para analizar la frase para despues añadirlo a un dataset. Estos son los pasos que hemos realizado para crear el modelo.

Hemos creado un buscador de sinonimos de palabras para despues penalizarlas o puntuarlas mejor.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/buscador_sinonimos.png)

En este codigo estamos haciendo una puntuacion a una frase comprobando si tiene palabras negativas para puntualizarla peor o si tiene palabras positivas para puntualizarlas con mayor puntaución.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/funcion_puntualizar_frases.png)

Vamos a comrobarlas individualmente haber si tiene un resultado optimo.

![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/comprobacion%20individual.png)

Hemos creado una funcion para puntualizar un array completo de textos y que me los muestre puntualizados.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/funcion_puntualizar_arrays.png)

Puntualizamos las frases que vamos a añadir al dataset.
Podemos ver que esta puntualizando correctamente las frases, estas frases se añadiria a un excel borrando las palabras individuales y borrando frases en blanco. Una vez que tengamos suficientes frases crearemos el dataset.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/puntualizar_frases_a%C3%B1adir_dataset.png)

Una vez creado el dataset esta es su informacion.
Tiene 343 registros, no tenemos valores nulos y tenemos dos columnas:
Texto: frase que se a puntualizado
Puntuacion: puntuación de la frase
Aqui podemos ver los primeros 5 registros del dataset y un describe al dataset que podemos ver que el valor minimo es un 0.01, el maximo 4,84 y la media 2,69. Tenemos unos buenos datos.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/muestra_de_los_datos.png)

Hemos creado varios modelos de prueba que se pueden ver en el cuaderno jupyter pero al final nos hemos quedado con un modelo de regresion lineal que es el que mejor resultado nos ha dado.
Podemos ver que tenemos un 92% de acierto en este modelo.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/modelo_de_regresion_lineal.png)

En esta imagen comparamos los modelos que hicimos para puntualizar algunas frases y ver cual es el modelo que es mas coherente.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/resultados_de_los_modelos.png)

Alfinal nos quedmaos con el que  mayor coerencia tiene pensamos que es el de LinearRegression.



## 8. Aplicación web
## 9. Conclusiones
