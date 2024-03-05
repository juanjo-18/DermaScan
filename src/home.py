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
class Home(HydraHeadApp):

# PEGAMOS NUESTRO CODIGO DE PAGINA 
    def run(self):
        

        st.markdown(f"<h1 style='text-align:center;'>DermaScan</h1>", unsafe_allow_html=True)

        # Encabezado principal
        st.markdown(f"<h2 style='text-align:center;'> ¡Bienvenidos a DermaScan!  -  Descubre el Futuro del Cuidado de la Piel</h2>", unsafe_allow_html=True)
        #st.markdown(f"<h3 style='text-align:center;'> Descubre el Futuro del Cuidado de la Piel </h3>", unsafe_allow_html=True)

        st.divider()

        col1, col2 = st.columns([1, 1])
        with col1:
        # Sección: DermaScan App
            st.subheader("DermaScan App: Tu Aliado en la Lucha Contra el Cáncer de Piel")
            st.image("imagenes/skin_scan2.jpg", use_column_width=True)
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>DermaScan App representa la vanguardia en el cuidado de la piel, utilizando modelos de inteligencia artificial para analizar imágenes y detectar posibles lesiones cutáneas.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Nuestra aplicación puede identificar si una lesión es maligna, así como clasificarla según su tipo, proporcionando a los usuarios una herramienta poderosa para la detección temprana y la prevención del cáncer de piel.</p>", unsafe_allow_html=True)
            st.page_link("./dermascan_app.py", label="App", icon="🤳")
        with col2:
        # Sección: Prevención y cuidado de la piel
            st.subheader("Prevención y Cuidado de la Piel: Tu Guía hacia una Piel Saludable")
            st.image("imagenes/skin_care.jpg", use_column_width=True)
            st.markdown(f"<p style='text-align:center;'>En esta sección, encontrarás una amplia gama de recomendaciones y consejos sobre cómo mantener tu piel saludable y protegida contra los daños solares.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Desde prácticas diarias de cuidado de la piel hasta medidas preventivas contra el cáncer de piel, estamos aquí para ayudarte a mantener una piel radiante y saludable en todo momento.</p>", unsafe_allow_html=True)
            
        st.divider()

        col3, col4 = st.columns([1, 1])
        with col3:
        # Sección: Incidencia Solar UV
            st.subheader("Incidencia Solar UV: Conoce el Impacto del Sol en tu Piel")
            st.image("imagenes/uv_index.jpg", use_column_width=True)
            st.markdown(f"<p style='text-align:center;'>Descubre la importancia de estar conscienciado del índice UV y su impacto en tu piel.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>En esta sección, te proporcionamos información en tiempo real sobre el índice UV actual en diferentes regiones, así como pronósticos para los próximos días. Mantente informado y protege tu piel contra los daños causados por la radiación solar.</p>", unsafe_allow_html=True)

        with col4:
        # Sección: About us
            st.subheader("About us: Conoce a los Creadores de DermaScan")
            st.image("imagenes/sun_1.jpg", use_column_width=True)
            st.markdown(f"<p style='text-align:center;'>Somos Juanjo Medina y Jesús Cánovas, dos apasionados del aprendizaje automático y la ciencia de datos.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Nuestra inspiración para desarrollar DermaScan surgió de la necesidad de crear un modelo de inteligencia artificial para la detección temprana de enfermedades de la piel. </p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Como residentes de Málaga, somos conscientes de la importancia de proteger la piel contra la radiación solar UV, lo que nos impulsó a crear esta aplicación innovadora como nuestro proyecto final de máster. Únete a nosotros en nuestra misión sobre el concienciamiento y la precvención del cáncer de piel.</p>", unsafe_allow_html=True)
