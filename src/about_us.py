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



class About_US(HydraHeadApp):

# PEGAMOS NUESTRO CODIGO DE PAGINA
    def run(self):
        
        
        st.markdown(f"<h1 style='text-align:center;'>Conoce a los Creadores de DermaScan</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>¡Bienvenido a nuestra sección 'Sobre nosotros'! Aquí puedes conocer más sobre los creadores detrás de DermaScan: Juanjo Medina y Jesús Cánovas.</h3>", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"<p style='text-align:center;'>Somos Juanjo Medina y Jesús Cánovas, dos apasionados del ML, DL y Data Science. Nos conocimos mientras realizábamos un Máster en AI y Big Data en la modalidad FP in Company en Accenture Málaga. </p>", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"<p style='text-align:center;'>Nuestra inspiración para desarrollar DermaScan surgió de la necesidad de crear un modelo AI para la detección temprana de enfermedades de la piel. </p>", unsafe_allow_html=True)
        st.divider()
        st.markdown(f"<p style='text-align:center;'>Como residentes de Málaga, somos conscientes de la importancia de proteger la piel contra la radiación solar UV, lo que nos impulsó a crear esta aplicación innovadora como nuestro Proyecto Fin de Máster. Únete a nosotros en nuestra misión de prevenir y concienciar del cáncer de piel.</p>", unsafe_allow_html=True)

         # ---- CONTACT ----

        # Use local CSS
        def local_css(file_name):
            with open(file_name) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        local_css("style/style.css")

        with st.container():
            
            # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
            contact_form = """
            <form action="https://formsubmit.co/stdust69cleaner@gmail.com" method="POST">
                <input type="hidden" name="_captcha" value="false">
                <input type="text" name="name" placeholder="Tu nombre" required>
                <input type="email" name="email" placeholder="Tu email" required>
                <textarea name="message" placeholder="Escribe tu mensaje aqui" required></textarea>
                <button type="submit">Enviar</button>
            </form>
            """
            col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 3, 1])


            with col2:
                st.markdown(f"<h2 style='text-align:center;'>Contáctanos!</h2>", unsafe_allow_html=True)
                st.markdown(contact_form, unsafe_allow_html=True)
                
            with col4:
                # COLUMNA DE RATINGS (20%)
       
                @st.cache_data
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

                @st.cache_data
                def make_prediction(review):
                    # Limpiamos los datos llamando a la función de limpieza
                    clean_review = text_cleaning(review)
                
                    # Cargamos el modelo
                    model = joblib.load("model/sentiment_MNB.pkl")
                
                    # Hacemos la predicción
                    result = model.predict([clean_review])
                
                    # Calculamos la probabilidad
                    probas = model.predict_proba([clean_review])
                    
                    return result, probas

            
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
                    
                        # Calcula la predicción
                        result, probability = make_prediction(texto_calificacion)
                        
                        #calificacion=modelo.predict([texto_calificacion])[0]
                        calificacion=(probability[0,1])*5
                        st.write("Tu calificacion a sido de: ",calificacion)
                        guardar_puntuacion_en_s3(texto_calificacion, calificacion)
                mostrar_datos_desde_s3()

        st.divider()
