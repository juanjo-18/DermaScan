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



# CREMOS UNA CLASE PARA LA PAGINA
class Prevencion(HydraHeadApp):

# PEGAMOS NUESTRO CODIGO DE PAGINA 
    def run(self):
        st.set_page_config(
            page_title="DermaScan",
            page_icon=":😎:",
            layout="wide",  # Ancho completo
        )
        st.title("DermaScan")

        st.header("Prevención y Cuidado de la Piel: Tu Guía hacia una Piel Saludable")
        st.write("¡Bienvenido a nuestra sección de Prevención y Cuidado de la Piel! Aquí encontrarás una completa guía para mantener tu piel saludable y protegida en todo momento. Desde consejos sobre rutinas de cuidado diario hasta recomendaciones sobre protección solar y hábitos alimenticios, estamos aquí para ayudarte a mantener una piel radiante y resistente a los daños solares. Explora nuestros recursos y descubre cómo adoptar prácticas de cuidado de la piel que promuevan la salud a largo plazo.")

        # División de la pantalla en columnas proporcionales
        col8, col9, col10 = st.columns([1, 1, 1])

        # Sección de Tiempo de Exposición
        with col8:
            st.subheader("Tiempo de Exposición al Sol")
            st.image("imagenes/sun_1.jpg", use_column_width=True) 
            st.write("Limita tu tiempo de exposición al sol, especialmente durante las horas pico de radiación solar.")
            st.write("Usa ropa protectora, sombreros de ala ancha y busca sombra para reducir la exposición directa al sol.")
            st.write("Aplica protector solar de amplio espectro con factor de protección solar (FPS) 30 o superior cada dos horas, o más frecuentemente si sudas o te mojas.")

        # Sección de Hidratación
        with col9:
            st.subheader("Hidratación")
            st.image("imagenes/water_1.jpg", use_column_width=True)
            st.write("Mantente bien hidratado bebiendo suficiente agua a lo largo del día, especialmente en climas cálidos o cuando haces ejercicio.")
            st.write("Considera el consumo de frutas y verduras con alto contenido de agua, como sandía, pepino y naranjas, para ayudar a mantener la hidratación.")
            st.write("Evita el exceso de cafeína y alcohol, ya que pueden tener un efecto deshidratante en el cuerpo.")

        # Sección de Dieta Saludable
        with col10:
            st.subheader("Dieta Saludable")
            st.image("imagenes/food_1.jpg", use_column_width=True)
            st.write("Prioriza una dieta rica en frutas, verduras, granos enteros y proteínas magras para obtener los nutrientes necesarios para una piel saludable.")
            st.write("Los alimentos ricos en antioxidantes, como las bayas, el té verde y las nueces, pueden ayudar a proteger la piel del daño causado por los radicales libres.")
            st.write("Evita los alimentos procesados, altos en azúcares añadidos y grasas saturadas, ya que pueden contribuir a problemas cutáneos como el acné y la inflamación.")

        # División de la pantalla en columnas proporcionales
        col11, col12, col13 = st.columns([1, 1, 1])

        # Sección de Protección Solar
        with col11:
            st.subheader("Protección Solar")
            st.image("imagenes/sun_2.jpg", use_column_width=True)
            st.write("Usa protector solar incluso en días nublados o cuando estés bajo la sombra, ya que los rayos UV pueden penetrar las nubes y reflejarse en superficies como la arena y el agua.")
            st.write("Elige un protector solar que ofrezca protección de amplio espectro contra los rayos UVA y UVB, y asegúrate de aplicarlo generosamente en todas las áreas expuestas de la piel.")
            st.write("No te olvides de proteger tus labios con un bálsamo labial con SPF y usar gafas de sol con protección UV para proteger tus ojos del daño solar.")
            

        # Sección de otra recomendación
        with col12:
            st.subheader("Visita al dermatólogo")
            st.image("imagenes/zderm_1.jpg", use_column_width=True)
            st.write("Programa revisiones periódicas con un dermatólogo para realizar un seguimiento de la salud de tu piel y detectar cualquier cambio o problema potencial de manera temprana.")
            st.write("El dermatólogo puede ofrecerte consejos personalizados sobre cómo cuidar y proteger tu piel, así como recomendaciones específicas de productos y tratamientos adecuados para tus necesidades individuales.")
            st.write("No subestimes la importancia de consultar regularmente a un dermatólogo, especialmente si tienes antecedentes de problemas cutáneos, exposición solar frecuente o cambios en la apariencia de lunares o manchas en la piel.")

        # Sección de otra recomendación
        with col13:
            st.subheader("Estilo de vida saludable")
            st.image("imagenes/sport_1.jpg", use_column_width=True)
            st.write("Adopta un estilo de vida saludable que incluya hábitos como una alimentación equilibrada, ejercicio regular y descanso adecuado para promover la salud general de tu piel.")
            st.write("Evita el tabaquismo y reduce el consumo de alcohol, ya que estos hábitos pueden afectar negativamente la salud de la piel y aumentar el riesgo de problemas cutáneos como el envejecimiento prematuro y el cáncer de piel.")
            st.write("Mantén un equilibrio entre el trabajo y el descanso, y encuentra formas de gestionar el estrés, ya que el estrés crónico puede contribuir a problemas cutáneos como el acné y la dermatitis.")