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
            st.markdown(f"<p style='text-align:center; '>DermaScan es una aplicación revolucionaria que utiliza modelos de Inteligencia Artificial</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; '>Entrenamos nuestros modelos con imágenes de lesiones piel etiquetadas por Dermatólogos.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; '>Nuestra aplicación es capaz de detectar la presencia de lesiones, determinar si son malignas o no y clasificarlas según su tipo. </p>", unsafe_allow_html=True)
            
            
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


        