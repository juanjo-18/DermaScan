import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import tensorflow
import keras
import boto3
import os
import spacy
import base64

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
        
        st.markdown(f"<h1 style='text-align:center;'> Cuidado y Prevención: Tu piel siempre saludable </h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align:center;'> ¡Bienvenido a nuestra sección de Prevención y Cuidado de la Piel! </h2>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center; margin-left:20%; margin-right:20%'>  Aquí encontrarás una sencilla guía para mantener tu piel saludable y protegida en todo momento. Desde consejos y recomendaciones para protegerte del sol y mantener una piel saludable. Descubre cómo adoptar prácticas de cuidado de la piel que promuevan la salud a largo plazo.</h3>", unsafe_allow_html=True)
        st.divider()

        col1, col2, col3, col4, col5, col6, col7= st.columns([0.5, 2.5, 0.5, 2.5, 0.5, 2.5, 0.5])

        with col2:
        # Sección: DermaScan App
            st.markdown(f"<h3 style='text-align:center;'> Exposición al Sol </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Limita tu tiempo de exposición al sol, especialmente durante las horas pico de radiación solar.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Usa ropa protectora, sombreros de ala ancha y busca sombra para reducir la exposición directa al sol.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Aplicate protección solar como mínimo SPF30, cada hora.</p>", unsafe_allow_html=True)
            st.divider()

            style_image1 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/sun1_prev.jpg" height="600" style="{style_image1}">', unsafe_allow_html=True)
            
    
        with col4:
        # Sección: Prevención y cuidado de la piel
            st.markdown(f"<h3 style='text-align:center;'>Hidratación </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Mantente bien hidratado bebiendo suficiente agua a lo largo del día, especialmente en climas cálidos o cuando haces ejercicio.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Incluye en tu dieta frutas y verduras con alto contenido de agua, como sandía, pepino y naranjas.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Evita el tabaco, el exceso de cafeína y  el alcohol</p>", unsafe_allow_html=True)
            st.divider()
            
            style_image2 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/agua2_prev.jpg" height="600" style="{style_image2}">', unsafe_allow_html=True)
            

        with col6:
        # Sección: Prevención y cuidado de la piel
            st.markdown(f"<h3 style='text-align:center;'>Dieta Saludable </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center;'>Intenta mantener una dieta equilibrada de frutas, verduras, proteínas y grasas saludables para obtener los nutrientes necesarios para una piel saludable.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Los alimentos ricos en antioxidantes, como las bayas, el té verde y las nueces, pueden ayudar a proteger la piel del daño causado por los radicales libres.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Evita los alimentos procesados, altos en azúcares añadidos y grasas saturadas.</p>", unsafe_allow_html=True)
            st.divider()

            style_image3 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/dieta3_prev.jpg" height="600" style="{style_image3}">', unsafe_allow_html=True)
            
        st.divider()

        col8, col9, col10, col11, col12, col13, col14= st.columns([0.5, 2.5, 0.5, 2.5, 0.5, 2.5, 0.5])

        with col9:
        # Sección: DermaScan App
            st.markdown(f"<h3 style='text-align:center;'> Protección Solar </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Usa protector solar incluso en días nublados o cuando estés bajo la sombra.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Elige un protector solar que ofrezca protección de amplio espectro contra los rayos UVA y UVB.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Protege tus labios con un bálsamo labial con SPF y usar gafas de sol con protección UV para proteger tus ojos del daño solar.</p>", unsafe_allow_html=True)
            st.divider()

            style_image4 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/protec4_prev.jpg" height="600" style="{style_image4}">', unsafe_allow_html=True)

    
        with col11:
        # Sección: Prevención y cuidado de la piel
            st.markdown(f"<h3 style='text-align:center;'>Visita al dermatólogo </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>Programa revisiones periódicas con un dermatólogo para favorecer la detección temprana de cualquier posible lesión.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>El dermatólogo proporciona consejos personalizados de cuidado y protección además de tratamientos adecuados para tus necesidades individuales.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Visita al Dermatólogo especialmente si tienes antecedentes de problemas cutáneos, exposición solar frecuente o cambios en la apariencia de lunares o manchas en la piel.</p>", unsafe_allow_html=True)
            st.divider()

            style_image5 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/demat5_prev.jpg" height="600" style="{style_image5}">', unsafe_allow_html=True)
            

        with col13:
        # Sección: Prevención y cuidado de la piel
            st.markdown(f"<h3 style='text-align:center;'>Estilo de vida saludable </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center;'>Adopta un estilo de vida saludable que incluya hábitos como una alimentación equilibrada, ejercicio regular y descanso adecuado para promover la salud general de tu piel.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>vita el tabaquismo y reduce el consumo de alcohol, ya que estos hábitos pueden afectar negativamente la salud de la piel y aumentar el riesgo de problemas cutáneos como el envejecimiento prematuro y el cáncer de piel.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Mantén un equilibrio entre el trabajo y el descanso, y encuentra formas de gestionar el estrés, ya que el estrés crónico puede contribuir a problemas cutáneos como el acné y la dermatitis.</p>", unsafe_allow_html=True)
            st.divider()

            style_image6 = """
            display: flow;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
            height: auto;
            max-width: 600px;
            max-height: 600px;
            justify-content: center;
            border-radius: 20%;
            """
            st.markdown(f'<img src="./app/static/depor6_prev.jpg" height="600" style="{style_image6}">', unsafe_allow_html=True)

        st.divider()
