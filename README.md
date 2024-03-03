
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

Primero guardamos todas las imagenes en una variable.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/renombrar_imagenes_declararlo.png)

Ahora vamos ha modificar el nombre de todas las imagenes a un nombre mas corto, ya que al tener un nombre mas corto el entrenamiento se realiza mas rapido.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/codigo_renombrar_imagenes.png)

En este bloque de codigo estamos buscando dentro de la misma carpeta si hay alguna imagen repetida, ya qu eso nos perjudicaria en el rendimiento del modelo.
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
carpeta_origen_benigno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno/benign'  
# Carpeta de origen con las imágenes
carpeta_origen_maligno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno/malignant'  

# Carpeta donde se guardarán las imágenes no duplicadas
carpeta_destino_benigno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno_sin_duplicados/bening'
# Carpeta donde se guardarán las imágenes no duplicadas
carpeta_destino_maligno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno_sin_duplicados/malignant'

if not os.path.exists(carpeta_destino):
    os.makedirs(carpeta_destino)

encontrar_duplicados(carpeta_origen, carpeta_destino)
</code>
</pre>

En este otro codigo estamos comprobando que no haya ninguna imagen en otra carpeta ya que eso significaria que esta mal etiquetada si esta en las dos categorias, por lo que si encuentra alguna las borra en los dos lados ya que no sabemos cual es la correcta.
<pre>
   <code class="language-python" id="codigo-ejemplo2">
def encontrar_duplicados(carpeta_origen1, carpeta_origen2, carpeta_destino1, carpeta_destino2):
    modelo_vgg16 = VGG16(weights='imagenet', include_top=False)

    imagenes1 = os.listdir(carpeta_origen1)
    imagenes2 = os.listdir(carpeta_origen2)

    caracteristicas1 = []
    caracteristicas2 = []

    duplicadas = []  # Almacena información sobre imágenes duplicadas

    for i, imagen in enumerate(imagenes1):
        ruta_imagen = os.path.join(carpeta_origen1, imagen)
        caracteristicas1.append(extraer_caracteristicas(modelo_vgg16, ruta_imagen))

    for j, imagen in enumerate(imagenes2):
        ruta_imagen = os.path.join(carpeta_origen2, imagen)
        caracteristicas2.append(extraer_caracteristicas(modelo_vgg16, ruta_imagen))

    caracteristicas1 = np.array(caracteristicas1)
    caracteristicas2 = np.array(caracteristicas2)

    similitud = cosine_similarity(caracteristicas1, caracteristicas2)

    umbral_similitud = 0.75  # Puedes ajustar este umbral según tus necesidades

    for i in range(len(imagenes1)):
        for j in range(len(imagenes2)):
            if similitud[i, j] > umbral_similitud:
                print(f"Imágenes duplicadas: {imagenes1[i]} y {imagenes2[j]}")
                duplicadas.append((imagenes1[i], imagenes2[j]))

    conteo_duplicadas = len(duplicadas)

    for imagen1, imagen2 in duplicadas:
        # Borrar imágenes duplicadas de ambas carpetas
        os.remove(os.path.join(carpeta_origen1, imagen1))
        os.remove(os.path.join(carpeta_origen2, imagen2))

    for i in range(len(imagenes1)):
        if imagenes1[i] not in [imagen1 for imagen1, _ in duplicadas]:
            shutil.copy(os.path.join(carpeta_origen1, imagenes1[i]), os.path.join(carpeta_destino1, imagenes1[i]))

    for j in range(len(imagenes2)):
        if imagenes2[j] not in [imagen2 for _, imagen2 in duplicadas]:
            shutil.copy(os.path.join(carpeta_origen2, imagenes2[j]), os.path.join(carpeta_destino2, imagenes2[j]))

    print(f"Total de imágenes duplicadas: {conteo_duplicadas}")
    print("Imágenes no duplicadas guardadas en las carpetas:", carpeta_destino1, carpeta_destino2)

# Carpeta de origen con las imágenes
carpeta_origen_benigno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno/benign'  
# Carpeta de origen con las imágenes
carpeta_origen_maligno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno/malignant' 

# Carpeta donde se guardarán las imágenes no duplicadas benignas
carpeta_destino_benigno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno_sin_duplicados/benign' 
# Carpeta donde se guardarán las imágenes no duplicadas malignas
carpeta_destino_maligno = 'C:/Users/juanj/prueba_de_imagenes_objetos/super_benigno_vs_maligno_sin_duplicados/malignant'  

# Asegurarse de que los directorios de destino existan, o créalos si no existen
if not os.path.exists(carpeta_destino_benigno):
    os.makedirs(carpeta_destino_benigno)

if not os.path.exists(carpeta_destino_maligno):
    os.makedirs(carpeta_destino_maligno)

# Encontrar duplicados en la carpeta benigna y copiar las no duplicadas
encontrar_duplicados(carpeta_origen_benigno, carpeta_origen_maligno, carpeta_destino_benigno, carpeta_destino_maligno)
</code>
</pre>


## 4. Exploración y visualización de los datos.


## 5. Preparación de los datos para Machine Learning.

En todos nuestros modelos de clasificación de imagenes hemos realizado estos pasos:
- Redimensionamiento de las imagenes a 150 x 150 pixeles, despues concatenamos todos los datos en arrays, ya que al tenerlos en numerico el modelo puede trabajar mejor con ellos y mejorar muchisimo la velocidad de entrenamiento.
<pre>
   <code class="language-python" id="codigo-ejemplo2">
def redimensionar_imagen(ruta, nuevo_tamano):
    imagen = Image.open(ruta)
    imagen_redimensionada = imagen.resize(nuevo_tamano)
    arreglo_pixeles_redimensionado = np.array(imagen_redimensionada)
    return arreglo_pixeles_redimensionado

# Definir el nuevo tamaño deseado
nuevo_tamano = (150, 150)

# Redimensionar las imágenes de entrenamiento benignas
datos_train_piel_redimensionados = [redimensionar_imagen(os.path.join(train_dir, 'benign', nombre_imagen), nuevo_tamano) for nombre_imagen in os.listdir(os.path.join(train_dir, 'benign'))]

# Redimensionar las imágenes de entrenamiento malignas
datos_train_objeto_redimensionados = [redimensionar_imagen(os.path.join(train_dir, 'malignant', nombre_imagen), nuevo_tamano) for nombre_imagen in os.listdir(os.path.join(train_dir, 'malignant'))]

# Asegurarse de que todas las imágenes tengan el mismo tamaño
datos_train_piel_redimensionados = np.array([img for img in datos_train_piel_redimensionados if img.shape == (150, 150, 3)])
datos_train_objeto_redimensionados = np.array([img for img in datos_train_objeto_redimensionados if img.shape == (150, 150, 3)])

# Concatenar los datos de ambas clases
datos_train_redimensionados = np.concatenate([datos_train_piel_redimensionados, datos_train_objeto_redimensionados], axis=0)

# Crear etiquetas correspondientes
etiquetas_train_redimensionadas = np.concatenate([np.zeros(len(datos_train_piel_redimensionados)), np.ones(len(datos_train_objeto_redimensionados))], axis=0)
</code>
</pre>

- Aqui estamos dividiendo los datos para tener los datos de entrenamiento y test y sus etiquetas correspondientes, tambien normalizamos los pixeles diviendo entre 255 y convertimos la etiquetas a un formato one-hot.
<pre>
   <code class="language-python" id="codigo-ejemplo">
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datos_train_redimensionados, etiquetas_train_redimensionadas, test_size=0.2, random_state=42)

# Normalizar los píxeles dividiendo por 255
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convertir etiquetas a formato one-hot
y_train_one_hot = to_categorical(y_train, num_classes=2)
y_test_one_hot = to_categorical(y_test, num_classes=2)
</code>
</pre>

- En el siguiente bloque de codigo lo que tenemos es una clase CustomLearningRateScheduler que lo que hace es a la hora del callback llama a esta clase para ajustar dinámicamente la tasa de aprendizaje durante el entrenamiento de un modelo, la metrica que se esta monitoreando es el Val_accuracy con una paciencia de 1 epoca. Eso significa que si el val_accuracy de una epoca a otra a desminuido se ejecuta la clase disminuyendo el learning rate dividiendolo entre /2 eso significa que si antes teniamos 0.0001 ahora tendriamos 0.0005.
- Tambien tenemos un checkpoint que va guardando el modelo cada vez que mejora el val_accuracy porque muchas veces pasa que durante el entrenamiento ha habido alguna epoca mejor que la ultima donde ha terminado el modelo su entrenamiento.
- Por ultimo se esta haciendo un ImageDataGenerator que se encarga de realizar aumento de datos y preprocesamiento para conjuntos de entrenamiento y prueba en un problema de clasificación de imágenes. Esto es especialmente útil para mejorar la capacidad de generalización del modelo al exponerlo a variaciones en los datos durante el entrenamiento. El aumento de datos ayuda a prevenir el sobreajuste al proporcionar más variabilidad en el conjunto de entrenamiento.

<pre>
   <code class="language-python" id="1">
class CustomLearningRateScheduler(Callback):
    def __init__(self, factor=0.5, patience=1, min_lr=1e-12):
        super(CustomLearningRateScheduler, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_val_accuracy = float('-inf')
        self.wait = 0
        self.current_lr = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.current_lr is None:
            self.current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy', 0)

        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.wait = 0
            # Guardar el modelo cuando la precisión en el conjunto de validación mejora
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.current_lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.current_lr = new_lr  # Actualizar el valor actual de la tasa de aprendizaje
                print(f'Reverting learning rate to {new_lr}')
                self.wait = 0

# Crear el callback CustomLearningRateScheduler
custom_lr_scheduler = CustomLearningRateScheduler(factor=0.5, patience=1, min_lr=1e-12)

# Callback para guardar el modelo con la mejor precisión en el conjunto de validación
model_checkpoint = ModelCheckpoint('best_model_checkpoint.h5', save_best_only=True, monitor='val_accuracy', mode='max')

# Aumento de datos para el conjunto de entrenamiento y prueba
datagen_train = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen_test = ImageDataGenerator()

datagen_train.fit(X_train)
datagen_test.fit(X_test)

# Convertir etiquetas a one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes=2)
y_test_one_hot = to_categorical(y_test, num_classes=2)

# Generador de imágenes aumentadas para el conjunto de entrenamiento
train_generator = datagen_train.flow(X_train, y_train_one_hot, batch_size=32)

# Generador de imágenes para el conjunto de prueba
test_generator = datagen_test.flow(X_test, y_test_one_hot, batch_size=32)
</code>
</pre>

## 6. Entrenamiento del modelo y comprobación del rendimiento.
Para el entrenamiento de nuestros modelos hemos utilizado modelos preentrenados hemos utilizado varios que se pueden ver en los colab aquí voy a mostrar el mejor de cada modelo.

### Modelo benigno o maligno
En este modelo se ha utilizado un modelo preentrenado VGG16.
Este modelo nos ha dado una precisión con los datos de entremiento del 0.95%

<pre>
   <code class="language-python" id="codigo-ejemplo">

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3), pooling='max')

# Agregar capas personalizadas
x = base_model.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

# Crear el modelo
model4 = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers[-10:]:
    layer.trainable = True

# Compilar el modelo con el optimizador Adam y el callback ReduceLROnPlateau
model4.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo utilizando el generador de datos aumentados para el conjunto de entrenamiento
history=model4.fit(train_generator, validation_data=test_generator, epochs=20, callbacks=[custom_lr_scheduler, model_checkpoint])
</code>
</pre>

- Aquí mostramos una grafica de los modelos provados y el que mejor resultados nos dio es el mostrado anteriormente.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/varos_modelo_grafica.png)

- Aquí podemos ver la grafica durante el entrenamiento del modelo, podemos ver que el val_accuraccy va un poco peor que el accuracy.
  
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/grafica_entrenamiento_benigno_maligno.png)

- Podemos ver aqui los valores resultantes de nuestro modelo con la clase de test, sus aciertos, fallos, val_accuracy, la confusion matrix y el classification report. Podemos ver que falla un poco mas en la segunda prediccion que serian fotos malignas con un resultado de 125 frente a las 90 fallos de prediccion de la clase benigna.
  
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/comprobacion_del_modelo_benigno_maligno_test.png)

- En esta imagen estamos mostrando cuales son las imagenes las cuales el modelo a fallado en etiquetarlas.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fallos_modelo_benigno_maligno.png)

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

