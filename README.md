
# DermaScan
Enlaces sobre nuestro trabajo.
- Modelo de sentimientos: https://github.com/juanjo-18/DermaScan/blob/main/colabs/modelo_sentimientos.ipynb
- Modelo de objeto y piel:
- Modelo de piel sana y piel con lesion:
- Modelo de benigno y maligno: https://github.com/juanjo-18/DermaScan/blob/main/colabs/benigno_vs_maligno.ipynb
- Web:
- PDF:
- Video:

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


## 2. Obtención de los datos.

### Modelo de objeto o imagen piel.
Para el modelo de objeto o imagen de piel, hemos hecho una combinación de tres conjuntos de imágenes: uno que contenía objetos, otro que contenía imágenes de piel con cáncer y otro con imágenes de piel sana. Los tres han sido obtenidos de Kaggle.
- Objetos y imagenes variadas: https://www.kaggle.com/datasets/greg115/various-tagged-images
- Benignos o malignos: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
- Pieles sanas: https://www.kaggle.com/datasets/thanatw/isic2019-modded

### Modelo de piel sana o piel con lesión.
Para el modelo de piel sana o piel con lesión, hemos hecho una combinación de dos conjuntos de imágenes: uno que contenía imagenes de pieles con cancer y otro que contenía imágenes de pieles sanas. Los dos han sido obtenidos de Kaggle.
- Pieles sanas: https://www.kaggle.com/datasets/thanatw/isic2019-modded
- Piel con cancer: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

### Modelo de benigno o maligno.
Para el modelo de benigno o maligno, hemos utilizado este dataset que ya contenia las imagenes benignas y malignas separadas ademas lo hemos juntado con otro conjunto de imagenes mas. Los dos han sido obtenidos de kaggle.
- Benignos o malignos1: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
- Benignos o malignos2: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign?select=train

### Modelo de sentimientos.
Para el modelo de sentimientos hemos cogido 500 lineas de este enlace de twitts en español y despues lo hemos procesado.
- Enlace: https://huggingface.co/datasets/pysentimiento/spanish-tweets

### Web Scrapping.
A traves de este enlace del tiempo he creado una tabla para mostrarla en la web haciendo web scrapping.
- Enlace: https://www.tutiempo.net/malaga.html?datos=detallados
Primero saco todo el html de la web y lo guardo en una variable.

![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/web_scrapin_guardar_url.png)

Aqui estoy creando un dataframe que me guarde el dia, la temperatura maxima, la temperatura minima y el indice UV.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/codigo_web_scrapping.png)

Aqui muestro en la web la tabla que hemos scrapeado.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/web_scrapin_resultado.png)


## 3. Limpieza de datos (Preprocesado).

### Modelo de sentimientos.
Para el modelo de sentimientos primero los twits que tenemos les estamos eliminadon valores y caracteres que no son necesarios, como los iconos, eliminar los @ y el texto asociado, eliminar # y su texto, eliminar urls y convertir todo a minusculas.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/Limpiar%20textos.png)

Aqui estamos eliminado de las frases las stopwords para despues pasarselo al modelo.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/quitamos_stopword.png)

### El resto de modelos.
Como estamos trabajando con imagenes necesitamos hacer varias comprobaciones y arreglos antes de poder utilizarla, ahora vamos a contar algunas cosas realizadas.
<pre>
   <code class="language-python" id="codigo-ejemplo">
def cargar_imagen(ruta):
    img = image.load_img(ruta, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extraer_caracteristicas(modelo, ruta):
    img_array = cargar_imagen(ruta)
    caracteristicas = modelo.predict(img_array)
    return caracteristicas.flatten()

def encontrar_duplicados(carpeta_origen, carpeta_destino):
    modelo_vgg16 = VGG16(weights='imagenet', include_top=False)

    imagenes = []
    caracteristicas = []

    for root, _, files in os.walk(carpeta_origen):
        for file in files:
            ruta_imagen = os.path.join(root, file)
            imagenes.append(file)
            caracteristicas.append(extraer_caracteristicas(modelo_vgg16, ruta_imagen))

    caracteristicas = np.array(caracteristicas)
    similitud = cosine_similarity(caracteristicas)

    umbral_similitud = 0.75  # Puedes ajustar este umbral según tus necesidades

    imagenes_no_duplicadas = []
    conteo_duplicadas = 0

    for i in range(len(imagenes)):
        duplicada = False
        for j in range(i + 1, len(imagenes)):
            if similitud[i, j] > umbral_similitud:
                print(f"Imágenes duplicadas: {imagenes[i]} y {imagenes[j]}")
                duplicada = True
                conteo_duplicadas += 1
                break

        if not duplicada:
            imagenes_no_duplicadas.append(imagenes[i])
            shutil.copy(os.path.join(carpeta_origen, imagenes[i]), os.path.join(carpeta_destino, imagenes[i]))

    print(f"Total de imágenes duplicadas: {conteo_duplicadas}")
    print("Imágenes no duplicadas guardadas en la carpeta:", carpeta_destino)

# Carpeta de origen con las imágenes
carpeta_origen_benigno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno/benign'  # Cambiar a la ruta en tu caso
# Carpeta de origen con las imágenes
carpeta_origen_maligno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno/malignant'  # Cambiar a la ruta en tu caso

# Carpeta donde se guardarán las imágenes no duplicadas
carpeta_destino_benigno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno_sin_duplicados/bening'  # Cambiar a la ruta en tu caso
# Carpeta donde se guardarán las imágenes no duplicadas
carpeta_destino_maligno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno_sin_duplicados/malignant'  # Cambiar a la ruta en tu caso

# Asegurarse de que el directorio de destino exista, o créalo si no existe
if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

encontrar_duplicados(carpeta_origen, carpeta_destino)
</code>
</pre>


## 4. Exploración y visualización de los datos.
## 5. Preparación de los datos para Machine Learning.
## 6. Entrenamiento del modelo y comprobación del rendimiento.
## 7. Procesamiento de Lenguaje Natural.


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

