
<h1 style='text-align:center;'>¡Bienvenido a DermaScan!</h1>

![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/app_logo.png)

Enlaces sobre nuestros colabs del trabajo.
- Modelo de sentimientos: https://github.com/juanjo-18/DermaScan/blob/main/colabs/modelo_sentimientos.ipynb
- Modelo de clasificación de objeto y piel: https://github.com/juanjo-18/DermaScan/blob/main/colabs/objeto_vs_piel.ipynb
- Modelo de clasifiación piel sana y piel con lesion: https://github.com/juanjo-18/DermaScan/blob/main/colabs/piel_sana_vs_cancer.ipynb
- Modelo de clasificación de lesiones o de cancer de piel: https://github.com/juanjo-18/DermaScan/blob/main/colabs/cancer_vs_otras_lesiones.ipynb
- Modelo de clasificación benigno y maligno: https://github.com/juanjo-18/DermaScan/blob/main/colabs/benigno_vs_maligno.ipynb
- Modelo de clasificación de 3 tipos de cancer malignos: https://github.com/juanjo-18/DermaScan/blob/main/colabs/clasificador_malignos_3types.ipynb
- Modelo de clasificación de 2 tipos de cancer benignos: https://github.com/juanjo-18/DermaScan/blob/main/colabs/clasificacion_benignos_2_tipos.ipynb
- Web: https://dermascan.streamlit.app/
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
10. [Bibliografía](#10-Bibliografía)

# 1. Justificación y descripción del proyecto

- Nuestro proyecto **DermaScan** es una plataforma web dedicada a la detección de tumores de piel a través de la carga de imágenes. En este espacio interactivo, los usuarios podrán cargar sus imágenes y se aplicarn una variedad de filtros especializados. Desde distinguir entre objetos y áreas de la piel hasta analizar la salud cutánea y detectar posibles tumores, nuestro sistema ofrece una experiencia completa.


- Con herramientas avanzadas, los usuarios podrán obtener resultados precisos sobre la naturaleza del tejido cutáneo en la imagen, identificando si es sano o presenta algún tipo de tumor. Además, ofrecemos la capacidad de diferenciar entre tumores benignos y malignos, brindando información crucial para la toma de decisiones médicas.

![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/scan_1.jpg)

- Este servicio va más allá al proporcionar detalles específicos sobre el tipo de tumor detectado, permitiendo a los usuarios obtener información detallada sobre su condición. Nuestra misión es hacer que la detección temprana de problemas de piel sea accesible y efectiva, brindando a los usuarios la tranquilidad y la información necesaria para tomar decisiones informadas sobre su salud cutánea. Bienvenido a una nueva era de cuidado personalizado y empoderamiento a través de la tecnología."

![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/skin_care.jpg)


# 2. Obtención de los datos.

## Modelo de objeto o imagen piel.
Para el modelo de objeto o imagen de piel, hemos hecho una combinación de tres conjuntos de imágenes: uno que contenía objetos, otro que contenía imágenes de piel con cáncer y otro con imágenes de piel sana. Los tres han sido obtenidos de Kaggle.
- Objetos y imagenes variadas: https://www.kaggle.com/datasets/greg115/various-tagged-images
- Benignos o malignos: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
- Pieles sanas: https://www.kaggle.com/datasets/thanatw/isic2019-modded

## Modelo de piel sana o piel con lesión.
Para el modelo de piel sana o piel con lesión, hemos hecho una combinación de dos conjuntos de imágenes: uno que contenía imagenes de pieles con cancer y otro que contenía imágenes de pieles sanas. Los dos han sido obtenidos de Kaggle.
- Pieles sanas: https://www.kaggle.com/datasets/thanatw/isic2019-modded
- Piel con cancer: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

## Modelo de benigno o maligno.
Para el modelo de benigno o maligno, hemos utilizado estos dataset que ya contenian las imagenes benignas y malignas separadas ademas lo hemos juntado con otro conjunto de imagenes mas. Los dos han sido obtenidos de kaggle.
- Benignos o malignos1: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
- Benignos o malignos2: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign?select=train
  

## Modelo de piel con cancer o piel con otra lesión
Para el modelo de imágenes de piel con cáncer y piel con otras lesiones, hemos hecho una combinación de tres datasets distintos conjuntos de imágenes: dos que contenía imagenes de pieles con cáncer y otro que contenía otro tipo de lesiones. Los tres han sido obtenidos de Kaggle:
- Benignos o malignos1: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign?select=train
- Benignos o malignos2: https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images
- Varios tipos de lesiones: https://www.kaggle.com/datasets/yashjaiswal4559/skin-disease
Aun que este último dataset tiene imagenes tanto de cáncer como de otras lesiones que no son cáncer, para el apartado de cáncer hemos cogido las que son cancer sólo: melanoma y carcinomas basales. (Ya que las imagenes están etiquetasdas por carpetas según su clase).

## Modelo de clasificación de 3 tipos de cancer maligno
Para el modelo, hemos utilizado este dataset que contenia varias carpetas con imagenes clasificadas de las cuales hemos cogido melanoma, basall cell carcinoma y Squamous cell carcinoma. El dataset ha sido obtenido de kagle.
- Enlace: https://www.kaggle.com/datasets/riyaelizashaju/isic-skin-disease-image-dataset-labelled
  
## Modelo de clasificación de 2 tipos de cancer benigno
Para el modelo, hemos utilizado este dataset que contenia varias carpetas con imagenes clasificadas de las cuales hemos cogido queratosis seborreica, dermatofibroma y Melanocytic nevus. El dataset ha sido obtenido de kagle.
- Enlace: https://www.kaggle.com/datasets/riyaelizashaju/isic-skin-disease-image-dataset-labelled
  
## Modelo de sentimientos.
Para el modelo de sentimientos hemos cogido 500 lineas de este enlace de twitts en español y despues lo hemos procesado.
- Enlace: https://huggingface.co/datasets/pysentimiento/spanish-tweets

## Web Scrapping.
A traves de este enlace del tiempo he creado una tabla para mostrarla en la web haciendo web scrapping.
- Enlace: https://www.tutiempo.net/malaga.html?datos=detallados
Primero saco todo el html de la web y lo guardo en una variable.

![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/web_scrapin_guardar_url.png)

Aqui estoy creando un dataframe que me guarde el dia, la temperatura maxima, la temperatura minima y el indice UV.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/codigo_web_scrapping.png)

Aqui muestro en la web la tabla que hemos scrapeado.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/web_scrapin_resultado.png)


# 3. Limpieza de datos (Preprocesado).

## Modelo de sentimientos.
Para el modelo de sentimientos primero los twits que tenemos les estamos eliminadon valores y caracteres que no son necesarios, como los iconos, eliminar los @ y el texto asociado, eliminar # y su texto, eliminar urls y convertir todo a minusculas.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/Limpiar%20textos.png)

Aqui estamos eliminado de las frases las stopwords para despues pasarselo al modelo.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/imagenes_sentimientos/quitamos_stopword.png)

## El resto de modelos.
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

# 4. Exploración y visualización de los datos.

- En este proceso hemos utilizado una dinámica estructurada y sistemática con el mismo patrón en casi todos cuadernos Jupyter en los que hemos creado y entrenado nuestros modelos y es la siguiente:

## Contenido del dataset

* Para mostrar el contenido del dataset hemos utilizado el siguiente bloque de código:

<pre>
   <code class="language-python" id="contenido-dataset">
     
# Numero de imagenes para cada clase
nums_train = {}
nums_val = {}
for s in cancer_y_otros:
    nums_train[s] = len(os.listdir(train_dir + '/' + s))
img_per_class_train = pd.DataFrame(nums_train.values(), index=nums_train.keys(), columns=["no. of images"])
print('Train data distribution :')
img_per_class_train
     
   </code>
</pre>
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/img_visual/4_1.png)

## Distribución de los datos

* Creamos una gráfica de barras para representar la distribución de las imágenes del dataset ssegún su clase.

<pre>
   <code class="language-python" id="distribucion-dataset">
     
plt.figure(figsize=(7,7))
plt.title('Distribución de los datos',fontsize=30)
plt.ylabel('Número de imágenes',fontsize=20)
plt.xlabel('Tipo de cáncer de piel',fontsize=20)

keys = list(nums_train.keys())
vals = list(nums_train.values())
sns.barplot(x=keys, y=vals)
     
   </code>
</pre>

![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/img_visual/4_2.png)

## Visualización de los datos

* Creamos una función para visualizar los datos

<pre>
   <code class="language-python" id="visual-func">
     
# Funcion para mostrar imagénes
train = ImageFolder(train_dir, transform=transforms.ToTensor())
def show_image(image, label):
    print("Label :" + train.classes[label] + "(" + str(label) + ")")
    return image.permute(1, 2, 0)
     
   </code>
</pre>

* Sacamos una muestra de una de las clases que tengamos, en este caso cáncer tipo *Basal Cell Carcinoma*

<pre>
   <code class="language-python" id="visual-type1">
     
# Directorio que contiene las imágenes
basal_cell_carcinoma_dir = os.path.join(train_dir, "basal_cell_carcinoma")

# Muestra 6 imágenes
fig, axs = plt.subplots(2, 3, figsize=(12, 10))
fig.tight_layout(pad=0)

# Iterar sobre las primeras 6 imágenes del directorio
for i in range(6):
    image_files = [file for file in os.listdir(basal_cell_carcinoma_dir) if file.endswith('.jpg')]
    if i < len(image_files):
        image_path = os.path.join(basal_cell_carcinoma_dir, image_files[i])
        image = Image.open(image_path)
        axs[i//3, i%3].imshow(image)
        axs[i//3, i%3].set_title(f'Imagen {i+1}')
    else:
        axs[i//3, i%3].axis('off')  # No mostrar ejes si no hay imagen para mostrar
plt.show()
     
   </code>
</pre>

![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/img_visual/4_3.png)

* Podemos ver otra muestra de otra clase diferente que tengamos, en este caso cáncer tipo *Melanoma*, para ello sólo tenemos que cambiar la parte del código que accdede al directorio donde se encuentran las imágenes a las que queremos acceder:

<pre>
   <code class="language-python" id="visual-type2">
     
# Directorio que contiene las imágenes
basal_cell_carcinoma_dir = os.path.join(train_dir, "basal_cell_carcinoma")
     
   </code>
</pre>

![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/img_visual/4_4.png)




# 5. Preparación de los datos para Machine Learning.

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

# 6. Entrenamiento del modelo y comprobación del rendimiento.
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

## Modelo objeto o piel.
Este modelo es una red neuronal convoluciona, nos ha dado una precisión con los datos de entremiento del 0.98%
<pre>
   <code class="language-python" id="2">
      # Crear modelo
      model1 = keras.models.Sequential([
       Conv2D(64, 3, input_shape=(150, 150, 3)),
       BatchNormalization(),
       Activation('relu'),
       Dropout(0.1),
       MaxPooling2D(),
   
       Conv2D(64, 3),
       BatchNormalization(),
       Activation('relu'),
       Dropout(0.15),
       MaxPooling2D(),
   
       Flatten(),
       Dense(300, activation='relu'),
       Dropout(0.5),
       Dense(2, activation='softmax')
   ])
   
   # Compilar el modelo
   model1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
   
   # Mostrar resumen del modelo
   model1.summary()
   
   # Entrenar el modelo con los nuevos datos
   history1 = model1.fit(
       X_train,
       y_train_one_hot,
       validation_data=(X_test, y_test_one_hot),
       epochs=15,
       batch_size=32  # Ajustar según tus necesidades
   )
</code>
</pre>
- Aquí mostramos una grafica de los modelos provados y el que mejor resultados nos dio fue el DenseNet121.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/objeto_vs_piel/comparacion_modelos.png)

- Aquí podemos ver la grafica durante el entrenamiento del modelo, el val_accuraccy va un siempre mejor que el accuracy.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/objeto_vs_piel/grafica_.png)

- Podemos ver aqui los valores resultantes de nuestro modelo con la clase de test, sus aciertos, fallos, val_accuracy, la confusion matrix y el classification report. Podemos ver que a fallado 1 vez en el la primera clase y a fallado 2 veces en la segunda clase es casi perfecto.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/objeto_vs_piel/aciertos.png)

- En esta imagen estamos mostrando cuales son las imagenes las cuales el modelo a fallado en etiquetarlas.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/objeto_vs_piel/fallos.png)

## Modelo piel sana o piel con lesión.
En este modelo se ha utilizado un modelo preentrenado MobileNetV2.
Este modelo nos ha dado una precisión con los datos de entremiento del 0.9731%
<pre>
   <code class="language-python" id="103">
   base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3), pooling='max')

for layer in base_model.layers[-10:]:
    layer.trainable = True

# Aumento de datos
datagen = ImageDataGenerator(
    rotation_range=40,           # Rango de rotación más amplio
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,          # Agregar volteo vertical
    fill_mode='nearest'
)

datagen.fit(X_train)

# Crear modelo
model3 = Sequential([
    base_model,
    Dense(512, activation='relu', kernel_regularizer='l2'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu', kernel_regularizer='l2'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 8 clases en tu caso
])


# Compilar el modelo
model3.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Mostrar resumen del modelo
model3.summary()

# Añadir Early Stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Entrenar el modelo con Early Stopping utilizando generadores de datos
history3 = model3.fit(
    datagen.flow(X_train, y_train_one_hot, batch_size=32),
    validation_data=(X_test, y_test_one_hot),
    epochs=30,
    callbacks=[early_stopping]
)
</code>
</pre>

- Podemos ver en esta gráfica, que de los tres modelos entrenados el mejor a sido el MobileNetV2 con un 97,31% de precision en los datos de test.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/piel_sana_vs_piel_lesion/comparacion_modelos.png)

- En la siguiente gráfica el val accuracy es peor que el accuracy y el val acurracy va dando muchos picos durante el entrenamiento.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/piel_sana_vs_piel_lesion/grafica.png)

- En esta imagen vemos que para la primera clase no ha fallado pero para la segunda ha tenido 9 fallos.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/piel_sana_vs_piel_lesion/aciertos.png)

- En esta imagen estamos mostrando cuales son las imagenes las cuales el modelo a fallado en etiquetarlas.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/piel_sana_vs_piel_lesion/fallos.png)



## Modelo piel con lesion o cancer.
En este modelo se ha utilizado un modelo preentrenado Xception.
Este modelo nos ha dado una precisión con los datos de entremiento del 0.97%
<pre>
   <code class="language-python" id="1002">
   
base_model = Xception(weights='imagenet', include_top=False, input_shape=(tamano, tamano, 3), pooling='max')

# Crear el callback CustomLearningRateScheduler
custom_lr_scheduler = CustomLearningRateScheduler(factor=0.5, patience=2, min_lr=1e-12)

# Callback para guardar el modelo con la mejor precisión en el conjunto de validación
#model_checkpoint = ModelCheckpoint('best_model_checkpoint7.h5', save_best_only=True, monitor='val_accuracy', mode='max')

# PARA LOCAL: cambia la ruta al directorio local donde deseas guardar el archivo .h5
nombre_archivo = 'best_model_checkpoint7.h5'
ruta_guardado_local = 'C:/Users/admin2/Desktop/dermascan_colabs/modelos/' + nombre_archivo

# Callback para guardar el modelo con la mejor precisión en el conjunto de validación en tu ordenador local
model_checkpoint = ModelCheckpoint(ruta_guardado_local, save_best_only=True, monitor='val_accuracy', mode='max')

# Agregar capas personalizadas
x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

# Crear el modelo
model7 = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers[-15:]:
    layer.trainable = True


# Compilar el modelo con el optimizador Adam y el callback ReduceLROnPlateau
model7.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Mostrar resumen del modelo
model7.summary()

# Entrenar el modelo utilizando el generador de datos aumentados para el conjunto de entrenamiento
history7=model7.fit(train_generator, validation_data=test_generator, epochs=25, callbacks=[custom_lr_scheduler, model_checkpoint])
</code>
</pre>

- Podemos ver en la grafica que el porcentaje de precision de val_acurracy es menor al accuracy.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/piel_lesion_vs_cancer/gafrica.png)

- En esta imagen de confusion matrix tenemos en la primera clase mas del doble de errores que en la segunda clase.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/piel_lesion_vs_cancer/aciertos.png)

- En esta imagen estamos mostrando cuales son las imagenes las cuales el modelo a fallado en etiquetarlas.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/piel_lesion_vs_cancer/errores.png)

## Modelo de clasifiación de 3 tipos de cancer malignos.
En este modelo se ha utilizado un modelo preentrenado Xception.
Este modelo nos ha dado una precisión con los datos de entremiento del 0.9445%.
<pre>
   <code class="language-python" id="108">
   base_model = Xception(weights='imagenet', include_top=False, input_shape=(tamano, tamano, 3), pooling='max')

# Crear el callback CustomLearningRateScheduler
custom_lr_scheduler = CustomLearningRateScheduler(factor=0.5, patience=2, min_lr=1e-12)

# PARA LOCAL: cambia la ruta al directorio local donde deseas guardar el archivo .h5
nombre_archivo = 'best_model_checkpoint8.h5'
ruta_guardado_local = 'C:/Users/admin2/Desktop/dermascan_colabs/modelos/' + nombre_archivo

# Callback para guardar el modelo con la mejor precisión en el conjunto de validación
model_checkpoint = ModelCheckpoint(ruta_guardado_local, save_best_only=True, monitor='val_accuracy', mode='max')

# Agregar capas personalizadas
x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

# Crear el modelo
model8 = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers[-10:]:
    layer.trainable = True

# Compilar el modelo con el optimizador Adam y el callback ReduceLROnPlateau
model8.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Mostrar resumen del modelo
model8.summary()

# Entrenar el modelo utilizando el generador de datos aumentados para el conjunto de entrenamiento
history=model8.fit(train_generator, validation_data=test_generator, epochs=20, callbacks=[custom_lr_scheduler, model_checkpoint])

# Continuar entrenando el modelo por más épocas
history_continued1 = model8.fit(train_generator, validation_data=test_generator, epochs=20, callbacks=[custom_lr_scheduler, model_checkpoint])

# Continuar entrenando el modelo por más épocas
history_continued1 = model8.fit(train_generator, validation_data=test_generator, epochs=20, callbacks=[custom_lr_scheduler, model_checkpoint])
</code>
</pre>

- En esta grafica el validation accuracy ha ido muy pegado al accuracy lo que es bueno.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/tipos_de_cancer_malignos/grafico.png)

- En esta imagen de confusion matrix tenemos mas fallos en la primera clase con 44 comparado a la segunda clase con 25 y la tercera con 25 tambien.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/tipos_de_cancer_malignos/aciertos.png)

- En esta imagen estamos mostrando cuales son las imagenes las cuales el modelo a fallado en etiquetarlas.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/tipos_de_cancer_malignos/errores.png)


## Modelo de clasificación de 2 tipos de cancer benignos.
En este modelo se ha utilizado un modelo preentrenado Xception.
Este modelo nos ha dado una precisión con los datos de entremiento del 0.9389%.
<pre>
   <code class="language-python" id="1005">
   base_model = Xception(weights='imagenet', include_top=False, input_shape=(tamano, tamano, 3), pooling='max')

# Crear el callback CustomLearningRateScheduler
custom_lr_scheduler = CustomLearningRateScheduler(factor=0.5, patience=2, min_lr=1e-12)

# Callback para guardar el modelo con la mejor precisión en el conjunto de validación
model_checkpoint = ModelCheckpoint('best_model_checkpoint7.h5', save_best_only=True, monitor='val_accuracy', mode='max')

# Agregar capas personalizadas
x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

# Crear el modelo
model7 = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers[-10:]:
    layer.trainable = True


# Compilar el modelo con el optimizador Adam y el callback ReduceLROnPlateau
model7.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])


# Entrenar el modelo utilizando el generador de datos aumentados para el conjunto de entrenamiento
history=model7.fit(train_generator, validation_data=test_generator, epochs=20,callbacks=[custom_lr_scheduler, model_checkpoint])
     
</code>
</pre>
- Podemos ver en la siguiente imagen de la grafica que el val_acurracy al inicio si esta pegado al accuracy pero durante el paso de las epocas se fue distanciando.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/tipos_de_cancer_benignos/grafico.png)

- En esta imagen estamos mostrando cuales son las imagenes las cuales el modelo a fallado en etiquetarlas.
![Descripción de la imagen](https://github.com/juanjo-18/DermaScan/blob/main/imagenes/imagenes_readmi/fotos_modelos/tipos_de_cancer_benignos/fallos.png)


# 7. Procesamiento de Lenguaje Natural.

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

Alfinal nos quedamos con el que  mayor coerencia tiene pensamos que es el de LinearRegression.


# 8. Aplicación web
# 9. Conclusiones
# 10. Bibliografía

