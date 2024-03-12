import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import tensorflow
import keras
import boto3
import os

from keras.models import load_model
from PIL import Image
from datetime import datetime, timedelta
from st_files_connection import FilesConnection
from botocore.exceptions import NoCredentialsError
from hydralit import HydraApp
from hydralit import HydraHeadApp


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expressio
import warnings
nltk.download('wordnet')
nltk.download('stopwords')
warnings.filterwarnings("ignore")

np.random.seed(123)
 
stop_words = stopwords.words("spanish")


# CREMOS UNA CLASE PARA LA PAGINA
class DermascanApp(HydraHeadApp):

# PEGAMOS NUESTRO CODIGO DE PAGINA 
    def run(self):
        
        calificacion = 0
        # Set the layout to two columns
        col5, col6,col7 = st.columns([6,1,3])
        with col5:
            st.markdown(f"<h1 style='text-align:center;'>¡Bienvenido a DermaScan App!</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align:center;'>Tu Aliada en la Lucha Contra el Cáncer de Piel</h2>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center; '>Nos situamos en la vanguardia de la tecnología del cuidado de la piel desarrollado una aplicación revolucionaria que utiliza modelos de Inteligencia Artificial</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; '>Entrenamos nuestros modelos con miles de imágenes de piel, tanto sana como afectada pos algún tipo de lesión. En total hemos entrenado 6 modelos clasificadores con casi 100.000 imágenes</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; '>Nuestra aplicación es capaz de detectar la presencia de lesiones, determinar si son malignas o no y clasificarlas según su tipo. </p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Con DermaScan App, puedes realizar evaluaciones rápidas y precisas de tu piel desde la comodidad de tu hogar, ayudando a detectar tempranamente posibles signos de cáncer de piel y promoviendo una piel más saludable y segura.</p>", unsafe_allow_html=True)
            
            st.divider()
            st.markdown(f"<h3 style='text-align:center;'>Sube o captura una imagen de tu piel</h3>", unsafe_allow_html=True)

            # Agregar un apartado para cargar una foto
            imagen = st.file_uploader("Carga la imagen", type=["jpg", "jpeg", "png"])

            # Verificar si se cargó una foto
            if imagen is not None:
                st.image(imagen, caption="Imagen cargada", use_column_width=True)
                
                try:
                    # Cargar los modelos
                    piel_vs_cancer = tf.lite.Interpreter(model_path='model/piel_sana_piel_cancer_mejorado.tflite')
                    benigno_vs_maligno = tf.lite.Interpreter(model_path='model/benigno_vs_maligno_mejorado2.tflite')
                    objeto_piel_modelo = tf.lite.Interpreter(model_path='model/piel_o_objeto_mejorado.tflite')

                    otra_lesion_vs_cancer=tf.lite.Interpreter(model_path='model/modelo_otras_lesiones_vs_cancer.tflite')
                    clasificador_benigno=tf.lite.Interpreter(model_path='model/clasificador_cancer_benignos.tflite')
                    clasificador_maligno=tf.lite.Interpreter(model_path='model/clasificador_cancer_malignos.tflite')
                except Exception as e:
                    st.error(f"Error al cargar el modelo: {str(e)}")
                
                # Convertir la imagen a un formato adecuado para la predicción
                imagen = Image.open(imagen).convert('RGB')
                imagen_reescalada=imagen.resize((150, 150))# Igualar al modelo original
                imagen_reescalada=np.array(imagen_reescalada)
                imagen_reescalada=imagen_reescalada / 255.0
                imagen_reescalada=np.expand_dims(imagen_reescalada.astype(np.float32), axis=0)

                
                try:
                    #  Realizar la predicción objeto vs piel
                    objeto_piel_modelo.allocate_tensors()
                    entrada_details2 = objeto_piel_modelo.get_input_details()
                    salida_details2 = objeto_piel_modelo.get_output_details()
                    objeto_piel_modelo.set_tensor(entrada_details2[0]['index'], imagen_reescalada)
                    objeto_piel_modelo.invoke()
                    prediccion_objeto_piel_modelo = objeto_piel_modelo.get_tensor(salida_details2[0]['index'])

                    #  Realizar la predicción piel vs cancer
                    piel_vs_cancer.allocate_tensors()
                    entrada_details = piel_vs_cancer.get_input_details()
                    salida_details = piel_vs_cancer.get_output_details()
                    piel_vs_cancer.set_tensor(entrada_details[0]['index'], imagen_reescalada)
                    piel_vs_cancer.invoke()
                    prediccion_piel_vs_cancer = piel_vs_cancer.get_tensor(salida_details[0]['index'])
                    
                    # Realizar la predicción de otra lesion o cancer
                    otra_lesion_vs_cancer.allocate_tensors()
                    entrada_details3 = otra_lesion_vs_cancer.get_input_details()
                    salida_details3 = otra_lesion_vs_cancer.get_output_details()
                    otra_lesion_vs_cancer.set_tensor(entrada_details3[0]['index'], imagen_reescalada)
                    otra_lesion_vs_cancer.invoke()
                    prediccion_otra_lesion_vs_cancer = otra_lesion_vs_cancer.get_tensor(salida_details3[0]['index'])

                    #  Realizar la predicción benigno maligno
                    benigno_vs_maligno.allocate_tensors()
                    entrada_details1 = benigno_vs_maligno.get_input_details()
                    salida_details1 = benigno_vs_maligno.get_output_details()
                    benigno_vs_maligno.set_tensor(entrada_details1[0]['index'], imagen_reescalada)
                    benigno_vs_maligno.invoke()
                    prediccion_benigno_vs_maligno = benigno_vs_maligno.get_tensor(salida_details1[0]['index'])

                    # Realizar predicción de clasificador benigno
                    clasificador_benigno.allocate_tensors()
                    entrada_details4 = clasificador_benigno.get_input_details()
                    salida_details4 = clasificador_benigno.get_output_details()
                    clasificador_benigno.set_tensor(entrada_details4[0]['index'], imagen_reescalada)
                    clasificador_benigno.invoke()
                    prediccion_clasificador_benigno = clasificador_benigno.get_tensor(salida_details4[0]['index'])

                    # Realizar predicción de clasificador maligno
                    clasificador_maligno.allocate_tensors()
                    entrada_details5 = clasificador_maligno.get_input_details()
                    salida_details5 = clasificador_maligno.get_output_details()
                    clasificador_maligno.set_tensor(entrada_details5[0]['index'], imagen_reescalada)
                    clasificador_maligno.invoke()
                    prediccion_clasificador_maligno = clasificador_maligno.get_tensor(salida_details5[0]['index'])


                    # Imprimir la predicción de objeto o piel
                    valor_prediccion_objeto_piel_modelo= np.argmax(prediccion_objeto_piel_modelo)
                    if valor_prediccion_objeto_piel_modelo == 1:
                        st.write("La imagen insertada no es piel.")
                    else:
                        # Imprimir la predicción de piel o piel cancer
                        clase_predicha =  np.argmax(prediccion_piel_vs_cancer)
                        if clase_predicha == 0:
                            st.write("La imagen es piel sana.")
                        else:
                            # Imprimir la predicción de otra lesion o cancer
                            clase_predicha = np.argmax(prediccion_otra_lesion_vs_cancer)
                            if clase_predicha == 1:
                               st.write("La imagen es otro tipo de lesión no cancerigena.")
                            else:
                                # Imprimir la predicción de benigna o maligna
                                clase_predicha = np.argmax(prediccion_benigno_vs_maligno)
                                if clase_predicha == 0:
                                    st.write("La imagen insertada es un cancer de piel benigno.")

                                    # Imprimir la predicción de clasificacion benigno
                                    clase_predicha = np.argmax(prediccion_clasificador_benigno)
                                    if clase_predicha == 0:
                                        st.write("")
                                    elif clase_predicha == 1:
                                        st.write("El tipo de cancer es Melanocytic nevus ")
                                    else:
                                        st.write("El tipo de cancer es Queratosis seborreica")
                                else:
                                    st.write("La imagen insertada es un cancer de piel maligno.")
                                    # Imprimir la predicción de clasificacion maligno
                                    clase_predicha = np.argmax(prediccion_clasificador_maligno)
                                    if clase_predicha == 0:
                                        st.write("El tipo de cancer es Basall cell carcinoma")
                                    elif clase_predicha == 1:
                                        st.write("El tipo de cancer es es Melanoma")
                                    else:
                                        st.write("El tipo de cancer es es Squamous cell carcinoma")
                            
                except Exception as e:
                    st.error(f"Error al hacer la prediccion: {str(e)}")


        # COLUMNA DE RATINGS (20%)
        with col7:
            def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
                # Clean the text
                text = re.sub(r"[^A-Za-z0-9]", " ", text) # eliminar caracteres especiales
                text = re.sub(r"\'s", " ", text) # quitar apóstrofes 
                text = re.sub(r"http\S+", " link ", text) # quitar enlaces sustituyendolos por "link"
                text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  
            
                # Quitamos signos de puntuacion
                text = "".join([c for c in text if c not in punctuation])
            
                # Quitamos Stopwords
                if remove_stop_words:
                    text = text.split()
                    text = [w for w in text if not w in stop_words]
                    text = " ".join(text)
            
                # Lemmatizamos las palabras
                if lemmatize_words:
                    text = text.split()
                    lemmatizer = WordNetLemmatizer()
                    lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
                    text = " ".join(lemmatized_words)
            
                # Devolvemos una lista de palabras tratadas
                return text

            # Llamar a la función para mostrar los datos desde S3
            def guardar_puntuacion_en_s3(comentario, puntuacion):
                # Convierte el comentario y la puntuación en un DataFrame
                df_nuevo = pd.DataFrame({"COMENTARIO": [comentario], "PUNTUACION": [puntuacion]})

                try:
                    # Conecta con S3 y lee el archivo existente
                    s3 = boto3.client('s3', aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"], aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
                    obj = s3.get_object(Bucket='dermascan-streamlits3', Key='pruebas3_streamlit.csv')
                    df_existente = pd.read_csv(obj['Body'])

                    # Concatena el nuevo DataFrame con el existente
                    df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)

                    # Escribe el DataFrame final en S3
                    s3.put_object(Bucket='dermascan-streamlits3', Key='pruebas3_streamlit.csv', Body=df_final.to_csv(index=False), ContentType='text/csv')
                except NoCredentialsError:
                    st.error("No se encontraron las credenciales de AWS. Por favor, configure sus credenciales correctamente.")

            
            def mostrar_datos_desde_s3():
                try:
                    # Conecta con S3 y lee el archivo CSV
                    s3 = boto3.client('s3', aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"], aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"])
                    obj = s3.get_object(Bucket='dermascan-streamlits3', Key='pruebas3_streamlit.csv')
                    df = pd.read_csv(obj['Body'])

                    suma_puntuaciones = df['PUNTUACION'].sum()
                    # Calcula la cantidad de registros en la columna 'puntuaciones'
                    cantidad_registros = len(df)

                    # Calcula el promedio dividiendo la suma total por la cantidad de registros
                    promedio = suma_puntuaciones / cantidad_registros
                    mostrar_imagen_segun_puntuacion(int(promedio))
                    st.markdown(f"<p style='text-align:center;'>¡La puntuación media es de {round(promedio,2)}!</p>", unsafe_allow_html=True)
                    
                    # Muestra el DataFrame en Streamlit
                    #st.write(df)

                    # Ordenar el DataFrame por puntuación en orden descendente
                    df_ordenado = df.sort_values(by='PUNTUACION', ascending=False)

                    # Extraer los tres mejores textos y asignarlos a variables separadas
                    comentario1 = df_ordenado.iloc[0]['COMENTARIO']
                    puntuacion1= round(df_ordenado.iloc[0]['PUNTUACION'],2)
                    comentario2 = df_ordenado.iloc[1]['COMENTARIO']
                    puntuacion2= round(df_ordenado.iloc[1]['PUNTUACION'],2)
                    comentario3 = df_ordenado.iloc[2]['COMENTARIO']
                    puntuacion3= round(df_ordenado.iloc[2]['PUNTUACION'],2)
                    
                    mejor_texto1 = f'1. "{comentario1}"  -  {puntuacion1}'
                    mejor_texto2 = f'2. "{comentario2}"  -  {puntuacion2}'
                    mejor_texto3 = f'3. "{comentario3}"  -  {puntuacion3}'

                    # Ordenar el DataFrame por puntuación en orden ascendente
                    df_ordenado = df.sort_values(by='PUNTUACION', ascending=True)

                    # Extraer los tres peores textos y asignarlos a variables separadas
                    comentario1 = df_ordenado.iloc[0]['COMENTARIO']
                    puntuacion1= round(df_ordenado.iloc[0]['PUNTUACION'],2)
                    comentario2 = df_ordenado.iloc[1]['COMENTARIO']
                    puntuacion2= round(df_ordenado.iloc[1]['PUNTUACION'],2)
                    comentario3 = df_ordenado.iloc[2]['COMENTARIO']
                    puntuacion3= round(df_ordenado.iloc[2]['PUNTUACION'],2)
                    
                    peor_texto1 = f'1. "{comentario1}"  -  {puntuacion1}'
                    peor_texto2 = f'2. "{comentario2}"  -  {puntuacion2}'
                    peor_texto3 = f'3. "{comentario3}"  -  {puntuacion3}'

                    # Crear un bloque vacío de 10px de altura usando HTML personalizado
                    bloque_vacio = '<div style="height: 20px;"></div>'
                    st.markdown(bloque_vacio, unsafe_allow_html=True)
                    crear_bloques_reseñas(mejor_texto1,mejor_texto2,mejor_texto3,"Las 3 mejores reseñas:")
                    st.markdown(bloque_vacio, unsafe_allow_html=True)
                    crear_bloques_reseñas(peor_texto1,peor_texto2,peor_texto3,"Las 3 peores reseñas:")
                    
                except NoCredentialsError:
                    st.error("No se encontraron las credenciales de AWS. Por favor, configure sus credenciales correctamente.")
            

            def crear_bloques_reseñas(texto,texto1,texto2,frase):
                # Concatenar los textos
                textos_concatenados = f"{texto}<br>{texto1}<br>{texto2}"

                estilo_bloque = (
                "background-color: white; "
                "color: black; "
                "padding: 10px; "
                "border-radius: 5px; "
                "box-shadow: 0 0 5px rgba(0, 0, 0, 0.1);"
                )
                # Mostrar el bloque en blanco con el texto encima
                st.write(frase)
                with st.container():
                    st.markdown('<div style="{}">{}</div>'.format(estilo_bloque, textos_concatenados), unsafe_allow_html=True)

            def make_prediction(review):
                # Limpiamos los datos llamando a la función de limpieza
                clean_review = text_cleaning(review)
            
                # Cargamos el modelo
                model = joblib.load("model/sentiment_SVC.pkl")
            
                # Hacemos la predicción
                result = model.predict([clean_review])
            
                # Calculamos la probabilidad
                #probas = model.predict_proba([clean_review])
                probas=result
                #probability = "{:.2f}".format(float(probas[:, result]))
            
                return result, probability

           

            modelo=joblib.load("model/sentimientos_modelo.pkl")  
            def mostrar_imagen_segun_puntuacion(puntuacion):
                if puntuacion == 0:
                    st.image("imagenes/estrellas_0.png", caption="", use_column_width=True)
                elif puntuacion == 1:
                    st.image("imagenes/estrellas_1.png", caption="", use_column_width=True)
                elif puntuacion == 2:
                    st.image("imagenes/estrellas_2.png", caption="", use_column_width=True)
                elif puntuacion == 3:
                    st.image("imagenes/estrellas_3.png", caption="", use_column_width=True)
                elif puntuacion == 4:
                    st.image("imagenes/estrellas_4.png", caption="", use_column_width=True)
                elif puntuacion == 5:
                    st.image("imagenes/estrellas_5.png", caption="", use_column_width=True)
            
            
            st.markdown(f"<h1 style='text-align:center;'>Valoraciones</h1>", unsafe_allow_html=True)
            #st.header("Valoraciones")
            # Casilla de entrada de texto
            st.markdown(f"<h3 style='text-align:center;'>Comenta que te ha parecido la App:</h3>", unsafe_allow_html=True)
            #st.write("Comenta que te ha parecido la App: ")
            # text_area para ingresar el comentario
            texto_calificacion = st.text_input("")
            if len(texto_calificacion.strip()) > 0:
                # Calcula la predicción
                result, probability = make_prediction(texto_calificacion)
        
                # Muestra los resultados 
                st.header("Resultados")
            
                if int(result) == 1:
                    st.write("Esto es una reseña positiva con una probabilidad del ", probability)
                else:
                    st.write("Esto es una reseña negativa con una probabilidad del ", probability)

            # Centra el botón utilizando st.button y estilo CSS
            button_html = """
                <style>
                    div.stButton > button {
                        width: 100%;
                        display: flex;
                        justify-content: center;
                    }
                </style>
            """
            st.markdown(button_html, unsafe_allow_html=True)

            # Agrega un botón para borrar el contenido del área de texto
            if st.button("Enviar"):
                    # Llamar a la función para guardar los datos en S3
                if len(texto_calificacion.strip()) > 0:
                    calificacion=modelo.predict([texto_calificacion])[0]
                    st.write("Tu calificacion a sido de: ",calificacion)
                    guardar_puntuacion_en_s3(texto_calificacion, calificacion)
            mostrar_datos_desde_s3()