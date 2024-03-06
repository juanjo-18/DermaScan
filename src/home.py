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

        with st.container():
            style_image1 = """
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 50%;
            height: auto;
            max-width: 350px;
            max-height: 350px;
            justify-content: center;
            border-radius: 20%;
            """

            col1, col2, col3 = st.columns(3)
            with col1:
                st.empty()
            with col2:
                st.markdown(f"<h1 style='text-align:center; font-size:50px;'>DermaScan</h1>", unsafe_allow_html=True)
                #st.image("imagenes/app_logo.png", use_column_width=True)
                st.markdown(f'<img src="./app/static/app_logo.png" height="333" style="{style_image1}">', unsafe_allow_html=True)
            with col2:
                st.empty()

        

        # Encabezado principal
        st.markdown(f"<h2 style='text-align:center;'> ¡Bienvenidos al Futuro del Cuidado de la Piel!</h2>", unsafe_allow_html=True)
        #st.markdown(f"<h3 style='text-align:center;'> Descubre el Futuro del Cuidado de la Piel </h3>", unsafe_allow_html=True)

        st.divider()

        col1, col2, col3, col4, col5 = st.columns([1, 3.5, 1, 3.5, 1])

        with col2:
        # Sección: DermaScan App
            
            st.markdown(f"<h3 style='text-align:center;'> DermaScan App: Tu Aliado en la Lucha Contra el Cáncer de Piel </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center; font-size:none; color:none;'>DermaScan App representa la vanguardia en el cuidado de la piel, utilizando modelos de inteligencia artificial para analizar imágenes y detectar posibles lesiones cutáneas.</p>", unsafe_allow_html=True)
            st.image("imagenes/scan_1.jpg", use_column_width=True)
            st.markdown(f"<p style='text-align:center;'>Nuestra aplicación puede identificar si una lesión es maligna y clasificarla por tipos, convirtiéndose en una potente herramienta para la detección temprana y la prevención del cáncer de piel.</p>", unsafe_allow_html=True)
    

        with col4:
        # Sección: Prevención y cuidado de la piel
            
            st.markdown(f"<h3 style='text-align:center;'> Prevención y Cuidado de la Piel: Tu Guía hacia una Piel Saludable </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center;'>En esta sección, encontrarás una amplia gama de recomendaciones y consejos sobre cómo mantener tu piel saludable y protegida contra los daños solares.</p>", unsafe_allow_html=True)
            st.image("imagenes/cuidado_sol.jpg", use_column_width=True)
            st.markdown(f"<p style='text-align:center;'>Desde prácticas diarias de cuidado de la piel hasta medidas preventivas contra el cáncer de piel, estamos aquí para ayudarte a mantener una piel radiante y saludable en todo momento.</p>", unsafe_allow_html=True)
            
        st.divider()

        col6, col7, col8, col9, col10 = st.columns([1, 3.5, 1, 3.5, 1])

        with col7:
        # Sección: Incidencia Solar UV
            
            st.markdown(f"<h3 style='text-align:center;'> Conoce el Impacto del Sol en tu Piel</h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center;'>Info en tiempo real sobre el índice UV actual en Málaga y pronósticos para los próximos días. </p>", unsafe_allow_html=True)
            st.image("imagenes/uv_index.jpg", use_column_width=True)
            st.markdown(f"<p style='text-align:center;'>Descubre la importancia de estar concienciado del índice UV y su impacto en tu piel.</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Mantente informado y protege tu piel contra los daños causados por la radiación solar.</p>", unsafe_allow_html=True)
        with col9:
        # Sección: About us
            
            st.markdown(f"<h3 style='text-align:center;'> ¡Conoce a los Creadores de DermaScan! </h3>", unsafe_allow_html=True)
            st.divider()
            st.markdown(f"<p style='text-align:center;'>Somos Juanjo Medina y Jesús Cánovas, dos apasionados del aprendizaje automático y la ciencia de datos.</p>", unsafe_allow_html=True)
            st.image("imagenes/us.jpg", use_column_width=True)
            st.markdown(f"<p style='text-align:center;'>La idea de desarrollar DermaScan surge de nuestra pasión por la tecnología y nuestra conciencia sobre los desafíos en la detección temprana de cáncer de piel. </p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>¡Únete a nosotros en nuestra misión de concienciar y prevenir el cáncer de piel mientras exploramos los límites de la AI y la salud digital!</p>", unsafe_allow_html=True)